"""v22 ranking harness: enumerate device placements for a working set and rank
them with the fsrr new 3-target predictor. Pure predictor call -- no re-run."""
import itertools, os, sys
sys.path.insert(0, "/home/msyu/PycharmProjects/fsrr-multimodel-scheduling")
import model_registry as reg
from deploy_predictor_logic import DeployPredictor, DEFAULT_ALPHA, DEFAULT_BETA

BUNDLE = {
    "cpu-gpu": "xgboost_model/artifacts/cpu_gpu/deploy_cpu_gpu",
    "cpu-npu": "xgboost_model/artifacts/cpu_npu/deploy_cpu_npu",
}
DEVS = {"cpu-gpu": ["cpu", "gpu"], "cpu-npu": ["cpu", "npu"]}

def allowed(model, dev, accel):
    # accel is 'gpu' or 'npu'; respect registry DEVICE_CONSTRAINTS
    ad = reg.allowed_devices(model, accelerators=[accel])
    return dev in ad

def enumerate_placements(working_set, platform):
    """working_set: list of dicts {model, infps, slo_ms, display}. Returns
    {combo_name: {viewkey: {model,execution,infps,slo_ms,display}}}."""
    accel = DEVS[platform][1]
    per_model_devs = []
    for w in working_set:
        opts = [d for d in DEVS[platform] if allowed(w["model"], d, accel)]
        per_model_devs.append(opts)
    combos = {}
    for i, assign in enumerate(itertools.product(*per_model_devs)):
        blob = {}
        for w, dev in zip(working_set, assign):
            key = w["display"]
            blob[key] = {"display": w["display"], "execution": dev,
                         "model": w["model"], "infps": w.get("infps"),
                         "slo_ms": w.get("slo_ms")}
        combos[f"combination_{i:04d}"] = blob
    return combos

def rank(working_set, platform, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    combos = enumerate_placements(working_set, platform)
    # vocabulary validation at generation time (task 92-2)
    names = {v["model"] for blob in combos.values() for v in blob.values()}
    bad = reg.unresolved_models(names)
    if bad:
        raise RuntimeError(f"ranking names unresolved models: {bad}")
    dp = DeployPredictor(log_callback=lambda m: None)
    best, df = dp.predict_best_combination(
        schedule_data=combos, model_input_path=BUNDLE[platform],
        alpha=alpha, beta=beta)
    df = df.sort_values("pred_score", ascending=False).reset_index(drop=True)
    return best, df, combos

if __name__ == "__main__":
    # determinism test working set: fg 4 + bg llama1b
    WS = [
        {"model": "yolo11s",      "display": "view1", "infps": 30, "slo_ms": 33},
        {"model": "yolo11m",      "display": "view2", "infps": 15, "slo_ms": 66},
        {"model": "resnet50",     "display": "view3", "infps": 60, "slo_ms": 11},
        {"model": "mobilenet_v2", "display": "view4", "infps": 90, "slo_ms": 8},
        {"model": "llama1b",      "display": "bg1",   "infps": 1,  "slo_ms": 2000},
    ]
    for plat in ["cpu-gpu", "cpu-npu"]:
        orders = []
        for rep in range(3):
            best, df, combos = rank(WS, plat)
            order = list(zip(df["combination"], df["pred_score"].round(6)))
            orders.append(order)
        same = all(o == orders[0] for o in orders)
        print(f"[{plat}] combos={len(combos)} best={best} "
              f"3x-identical-order={same}")
        if not same:
            for r,o in enumerate(orders):
                print(f"  rep{r}: {o[:5]}")
        else:
            print(f"  top5: {orders[0][:5]}")
