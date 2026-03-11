source .venv/bin/activate
rm -f xgboost_model/artifacts/gpu/xgb_model_x3_score_alpha_*.json
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  python xgboost_model/deploy_selector_xgb_suite.py train \
    --train_mode score \
    --perf_csv xgboost_model/dataset/gpu/train_x3.csv \
    --schedule_csv xgboost_model/dataset/gpu/train_schedules_x3.csv \
    --model_out xgboost_model/artifacts/gpu/xgb_model_x3_score_alpha_${alpha} \
    --alpha ${alpha}
done
python scripts/xgb_alpha_trained_sweep.py
python scripts/plot_alpha_comparison.py