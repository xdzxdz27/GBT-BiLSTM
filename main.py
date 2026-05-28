import pandas as pd
import os
from baselines_runner import run_baseline
from runner import run_gbfusion_experiment
import numpy as np

DATA_NAMES = ['AMZN','AAPL','MSFT','TSLA','VOWDE','Sony','Tencent','CrudeOil','BABA']

B2_CONFIG = {   
    "run_id": "B2-Full",
    "use_radius_pe": True,     # GB-PE Change
}


ABLATION_CONFIGS = [

    B2_CONFIG, # B2

    {

    },

    # no radius positional encoding
    { **B2_CONFIG,
        "run_id": "A11-NoRadiusPE",
        "use_radius_pe": False
    },

    #  Mean Pooling of Ball Features
    { **B2_CONFIG,
        "run_id": "A10-LinearMeanPool",
        "global_encoder": "mean_only",
        "use_radius_pe": False
    },
]

def main():
    all_results = []
    
    # BASELINES
    print("="*80)
    print("RUNNING: B1 Baseline Experiments (LSTM / GRU / etc.)")
    print("="*80)
    baseline_df = run_baseline(DATA_NAMES)
    if not baseline_df.empty:
        all_results.append(baseline_df)
    
#     # GBFusion Experiments
    print("\n" + "="*80)
    print("RUNNING: B2 & Ablation Experiments (GBFusion)")
    print("="*80)

    N_EPOCHS = 100 
    
    for config in ABLATION_CONFIGS:
        for data_name in DATA_NAMES:
            try:
                result_dict = run_gbfusion_experiment(
                    data_name=data_name,
                    n_epochs=N_EPOCHS,
                    **config,
                )
                if result_dict:
                    all_results.append(pd.DataFrame([result_dict]))
                    
            except Exception as e:
                print(f"!!! ERROR running [{config['run_id']}] on [{data_name}]: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE. Processing Results...")
    print("="*80)
    
    if not all_results:
        print("!!! No results generated. Exiting.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    
    final_df['Dataset'] = pd.Categorical(final_df['Dataset'], categories=DATA_NAMES, ordered=True)
    
    #  'Model_Rank'
    def get_model_rank(model_name):
        if model_name == 'B2-Full':
            return 100 
        elif 'A' in model_name and '-' in model_name: 
            return 50 
        else:
            return 0   
            
    final_df['Rank'] = final_df['Model'].apply(get_model_rank)
    
    final_df.sort_values(by=['Dataset', 'Rank', 'Model'], inplace=True)
    
    final_df.drop(columns=['Rank'], inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    
    # =========================================================
    
    os.makedirs("results", exist_ok=True)
    output_path = "results/A2_And_Tide.xlsx"
    
    final_df.to_excel(output_path, index=False)
    
    print(f"Successfully saved sorted results to: {output_path}")
    print("\n--- FINAL RESULTS PREVIEW (Grouped by Dataset) ---")
    print(final_df[['Dataset', 'Model', 'MAE', 'RMSE', 'R2']].head(20))


if __name__ == "__main__":
    main()
