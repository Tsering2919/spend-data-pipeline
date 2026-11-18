import yaml
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch

class SpendPipeline:
    def __init__(self, config_path, data_dir="data"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.company_id = self.config['company_id']
        self.currency = self.config['reporting_currency']
        self.data_dir = data_dir
        self.raw_data = {}
        self.clean_data = {}
        self.master_df = None
        self.taxonomy = []
        self.fx_df = None

    def ingest_data(self):
        print(f"\n========================================")
        print(f"--- Ingesting Data for Company {self.company_id} ---")
        for file_key, file_info in self.config['files'].items():
            file_path = os.path.join(self.data_dir, file_info['path'])
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                df.columns = df.columns.str.strip()
                self.raw_data[file_key] = df
                print(f"Loaded {file_key}: {len(df)} rows")
            
        tax_path = os.path.join(self.data_dir, self.config['taxonomy_file'])
        if os.path.exists(tax_path):
            self.tax_df = pd.read_csv(tax_path)
            self.taxonomy = self.tax_df.fillna('').agg(' -> '.join, axis=1).tolist()
        else:
            self.taxonomy = ["IT Hardware", "Maintenance", "Logistics", "Office Supplies"]

        # Load FX Rates (Required for Company B)
        fx_path = os.path.join(self.data_dir, 'fx_rates_usd.csv')
        if os.path.exists(fx_path):
            self.fx_df = pd.read_csv(fx_path)
            self.fx_df['month'] = self.fx_df['month'].astype(str).str.strip()

    def standardize_data(self):
        print("--- Standardizing Columns & Cleaning Spend ---")
        for file_key, file_info in self.config['files'].items():
            raw_df = self.raw_data[file_key].copy()
            mappings = file_info['mappings']
            clean_df = pd.DataFrame()
            
            # Rule 6: Prefix IDs for A, raw IDs for B & C
            if self.company_id == "A":
                prefix = "ERP:" if file_key == 'erp' else "TOOL:"
                clean_df['record_id'] = prefix + raw_df[mappings['record_id']].astype(str)
            else:
                clean_df['record_id'] = raw_df[mappings['record_id']].astype(str)
            
            clean_df['supplier_name'] = raw_df[mappings['supplier_name']].fillna("UNKNOWN")
            
            desc_cols = mappings['description']
            clean_df['description'] = raw_df[desc_cols].fillna('').agg(' '.join, axis=1)
            
            if mappings.get('date'):
                clean_df['date'] = pd.to_datetime(raw_df[mappings['date']], errors='coerce', format='mixed')
                
            if mappings.get('status'):
                clean_df['status'] = raw_df[mappings['status']].astype(str).str.lower()
            else:
                clean_df['status'] = 'approved'
                
            if mappings.get('spend_net'):
                clean_str = raw_df[mappings['spend_net']].astype(str).str.replace(',', '', regex=False)
                clean_df['spend'] = pd.to_numeric(clean_str, errors='coerce')
            else:
                gross_str = raw_df[mappings['spend_gross']].astype(str).str.replace(',', '', regex=False)
                gross = pd.to_numeric(gross_str, errors='coerce')
                tax_str = raw_df[mappings['tax_rate']].astype(str).str.replace('%', '', regex=False)
                tax_rate = pd.to_numeric(tax_str, errors='coerce')
                tax_rate = tax_rate.apply(lambda x: x/100 if pd.notnull(x) and x > 1 else x).fillna(0.15)
                clean_df['spend'] = gross / (1 + tax_rate)

            # --- FX CURRENCY CONVERSION ---
            if mappings.get('currency') and self.currency == 'USD':
                currency_col = raw_df[mappings['currency']].fillna('USD')
                rate_col = pd.to_numeric(raw_df.get(mappings.get('currency_rate', '')), errors='coerce')
                
                # Default native USD to 1.0
                rate_col = rate_col.mask(currency_col == 'USD', 1.0)
                
                # Lookup missing rates based on invoice month
                if rate_col.isna().any() and self.fx_df is not None:
                    month_str = clean_df['date'].dt.strftime('%Y-%m')
                    temp_df = pd.DataFrame({'currency': currency_col, 'month': month_str})
                    temp_df = temp_df.merge(self.fx_df, on=['currency', 'month'], how='left')
                    rate_col = rate_col.fillna(temp_df['units_per_usd'])
                    
                rate_col = rate_col.fillna(1.0) # Fallback safety
                clean_df['spend'] = clean_df['spend'] / rate_col
            
            if mappings.get('po_reference'):
                clean_df['po_reference'] = raw_df[mappings['po_reference']]
            else:
                clean_df['po_reference'] = raw_df[mappings['record_id']] 
            
            clean_df['source_system'] = file_key
            self.clean_data[file_key] = clean_df

    def combine_and_exclude(self):
        print("--- Combining Systems and Applying Exclusions ---")
        self.master_df = pd.concat(self.clean_data.values(), ignore_index=True)
        self.master_df['include'] = 1
        self.master_df['exclusion_reason'] = ""
        
        bad_status = ~self.master_df['status'].isin(['approved', 'paid'])
        self.master_df.loc[bad_status, 'include'] = 0
        self.master_df.loc[bad_status, 'exclusion_reason'] = 'NOT_APPROVED'
        
        cancelled = self.master_df['description'].str.lower().str.contains('cancel', na=False)
        self.master_df.loc[cancelled & (self.master_df['include'] == 1), ['include', 'exclusion_reason']] = [0, 'CANCELLED']

        non_proc_pattern = r'tax|intercompany|royalt|partner|financ'
        non_proc = self.master_df['description'].str.lower().str.contains(non_proc_pattern, na=False, regex=True)
        self.master_df.loc[non_proc & (self.master_df['include'] == 1), ['include', 'exclusion_reason']] = [0, 'NON_PROCUREMENT']

    def deduplicate_spend(self):
        print("--- Running Cross-System & Key Deduplication ---")
        self.master_df['match_key'] = self.master_df['po_reference'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.upper()
        
        # Cross-system deduplication only triggers if multiple files exist
        if 'erp' in self.clean_data and 'invoice_tool' in self.clean_data:
            valid_rows = self.master_df[self.master_df['include'] == 1]
            erp_keys = set(valid_rows[valid_rows['source_system'] == 'erp']['match_key'])
            tool_keys = set(valid_rows[valid_rows['source_system'] == 'invoice_tool']['match_key'])
            
            overlapping_keys = {k for k in erp_keys.intersection(tool_keys) if k and k != 'NAN'}
            is_duplicate = (self.master_df['source_system'] == 'invoice_tool') & (self.master_df['match_key'].isin(overlapping_keys))
            self.master_df.loc[is_duplicate, ['include', 'exclusion_reason']] = [0, 'DUPLICATE']

        supp_clean = self.master_df['supplier_name'].astype(str).str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
        self.master_df['supplier_key'] = supp_clean.str.replace(r'\s+(inc|ltd|llc|corp|co|gmbh|sa)$', '', regex=True).str.strip()
        self.master_df['item_key'] = self.master_df['description'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    def categorize_spend(self):
        print("--- Running Zero-Shot AI Classification ---")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        category_embeddings = model.encode(self.taxonomy, convert_to_tensor=True)
        valid_idx = self.master_df['include'] == 1
        
        texts_to_predict = (self.master_df.loc[valid_idx, 'supplier_name'].astype(str) + " : " + self.master_df.loc[valid_idx, 'description'].astype(str)).tolist()
        
        if not texts_to_predict:
            return
            
        spend_embeddings = model.encode(texts_to_predict, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
        cosine_scores = util.cos_sim(spend_embeddings, category_embeddings)
        best_match_indices = torch.argmax(cosine_scores, dim=1)
        confidences = torch.max(cosine_scores, dim=1).values.tolist()
        
        predicted_full_paths = [self.taxonomy[idx] for idx in best_match_indices]
        self.master_df.loc[valid_idx, 'predicted_full'] = predicted_full_paths
        self.master_df.loc[valid_idx, 'confidence'] = confidences
        
        splits = self.master_df.loc[valid_idx, 'predicted_full'].str.split(' -> ', expand=True)
        for col_idx, col_name in enumerate(['l1', 'l2', 'l3', 'l4']):
            if col_idx < splits.shape[1]:
                self.master_df.loc[valid_idx, col_name] = splits[col_idx]
            else:
                self.master_df.loc[valid_idx, col_name] = ""

    def export_results(self):
        print("--- Exporting Final Results ---")
        self.master_df.loc[self.master_df['include'] == 0, 'spend'] = 0.0
        
        output_cols = [
            'record_id', 'include', 'exclusion_reason', 
            'supplier_key', 'item_key', 'spend', 
            'l1', 'l2', 'l3', 'l4', 'confidence'
        ]
        
        for col in output_cols:
            if col not in self.master_df.columns:
                self.master_df[col] = ""
                
        final_output = self.master_df[output_cols]
        output_path = os.path.join(self.data_dir, f"Tsering_{self.company_id}.csv")
        final_output.to_csv(output_path, index=False)
        print(f"Pipeline Complete! Saved to: {output_path}")

# --- Execution Block ---
if __name__ == "__main__":
    configs = ["configs/company_A.yaml", "configs/company_B.yaml"]
    for config_path in configs:
        if os.path.exists(config_path):
            pipeline = SpendPipeline(config_path=config_path)
            pipeline.ingest_data()
            pipeline.standardize_data()
            pipeline.combine_and_exclude()
            pipeline.deduplicate_spend()
            pipeline.categorize_spend()
            pipeline.export_results()
        else:
            print(f"\nERROR: I am trying to run Company B, but I cannot find '{config_path}'!")