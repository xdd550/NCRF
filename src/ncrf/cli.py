"""Stable user-facing entry points; no server run is started by default."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import METHODS, load_config, project_root


def parser():
    p=argparse.ArgumentParser(description="NCRF: main model, fair comparators and component ablations")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("list",help="List registered methods and paper labels")
    plan=sub.add_parser("plan",help="Print the fixed training schedule; do not run")
    plan.add_argument("--method",choices=METHODS,default="ncrf")
    smoke=sub.add_parser("smoke",help="Synthetic CPU forward/backward check; not scientific results")
    smoke.add_argument("--method",choices=(*METHODS,"all"),default="all")
    preflight=sub.add_parser("preflight",help="Actual CUDA optimizer updates on real train data, not a full run")
    preflight.add_argument("--method",choices=(*METHODS,"all"),default="all")
    preflight.add_argument("--data-root")
    preflight.add_argument("--output",required=True)
    preflight.add_argument("--horizons",nargs="+",type=int,choices=(10,90),default=[10,90])
    check=sub.add_parser("validate-data",help="Verify frozen manifest, array hashes and grouping")
    check.add_argument("--data-root")
    check.add_argument("--split",choices=("train","valid","test"),default="valid")
    check.add_argument("--authorize-test",action="store_true")
    tr=sub.add_parser("train",help="Opt-in CUDA training with the registered fixed budget")
    tr.add_argument("--method",choices=[m for m in METHODS if m!="persistence"],required=True)
    tr.add_argument("--seed",type=int,required=True)
    tr.add_argument("--data-root")
    tr.add_argument("--output",required=True)
    tr.add_argument("--resume",action="store_true")
    tr.add_argument("--execute",action="store_true")
    tr.add_argument("--strict-determinism",action="store_true",help="New strict numerical mode; not the historical paper backend")
    for command in ("evaluate","benchmark"):
        q=sub.add_parser(command)
        q.add_argument("--method",choices=METHODS,default="ncrf")
        q.add_argument("--checkpoint")
        q.add_argument("--data-root")
        q.add_argument("--output",required=True)
        q.add_argument("--device",default="cuda")
        q.add_argument("--split",choices=("valid","test","ood") if command=="evaluate" else ("valid","test"),default="valid")
        q.add_argument("--authorize-test",action="store_true")
        if command=="evaluate":
            q.add_argument("--horizons",nargs="+",type=int,default=[10,20,50,90])
            q.add_argument("--limit",type=int)
            q.add_argument("--save-predictions",action=argparse.BooleanOptionalAction,default=True,
                           help="Save all evaluated trajectories by default; --no-save-predictions disables array saving")
            q.add_argument("--scenario",default="main")
            q.add_argument("--ood-array")
            q.add_argument("--ood-sha256")
            q.add_argument("--ood-metadata")
        else:
            q.add_argument("--trajectories",type=int,default=10)
            q.add_argument("--warmup",type=int,default=10)
            q.add_argument("--solver",action="store_true",help="Also rerun solver timing; omitted by default")
    summary=sub.add_parser("summarize")
    summary.add_argument("directories",nargs="+")
    summary.add_argument("--output",required=True)
    summary.add_argument("--allow-diagnostic",action="store_true")
    plot=sub.add_parser("plot")
    plot.add_argument("prediction",help="NPZ produced by evaluate --save-predictions")
    plot.add_argument("--output",required=True,help="New SVG/PNG/PDF path")
    return p


def synthetic_smoke(methods):
    import torch
    from .models.registry import build_model, parameter_count
    from .losses.long_horizon import VCHV15NeumannLongHorizonLoss
    from .utils import configure_backend,seed_all
    configure_backend()
    rows=[]
    for method in methods:
        seed_all(17)
        model=build_model(method,image_size=16,checkpoint_steps=False)
        inputs=(torch.randn(1,10,1,16,16)*.1,torch.randn(1,10,1,16,16),torch.randn(1,2,1,16,16))
        output=model(*inputs)
        if output.shape!=(1,2,1,16,16) or not bool(torch.isfinite(output).all()) or float(output.abs().max().detach())>1:
            raise RuntimeError(f"Invalid synthetic output: {method}")
        if method!="persistence":
            loss,_=VCHV15NeumannLongHorizonLoss(**load_config(method)["loss"])(output,torch.zeros_like(output))
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError(f"Nonfinite synthetic gradient: {method}")
        rows.append(dict(method=method,parameters=parameter_count(model),shape=list(output.shape),synthetic_only=True))
    return rows


def main(argv=None):
    args=parser().parse_args(argv)
    command=args.command
    options=vars(args).copy(); options.pop("command")
    if command=="list":
        result=[{k:v for k,v in load_config(m).items() if k in ("method","category","paper_label","description","expected_parameters")} for m in METHODS]
    elif command=="plan":
        result=load_config(args.method)
    elif command=="smoke":
        result=synthetic_smoke(METHODS if args.method=="all" else [args.method])
    elif command=="preflight":
        from .training.preflight import preflight
        result=preflight(METHODS if args.method=="all" else [args.method],args.data_root,args.output,horizons=args.horizons)
    elif command=="validate-data":
        from .data.dataset import data_root,open_split,metadata
        if args.split=="test" and not args.authorize_test:
            raise ValueError("Test access requires --authorize-test.")
        config=load_config("ncrf"); root=data_root(args.data_root)
        array,mean,std=open_split(root,config,args.split)
        records=metadata(root,config,args.split)
        result=dict(split=args.split,shape=list(array.shape),initial_fields=len({r["initial_id"] for r in records.values()}),mean=mean,std=std)
    elif command=="train":
        from .training.engine import train
        options["root"]=options.pop("data_root")
        result=train(**options)
    elif command in ("evaluate","benchmark"):
        options["root"]=options.pop("data_root")
        if command=="evaluate":
            from .evaluation.runner import evaluate
            result=evaluate(**options)
        else:
            from .evaluation.benchmark import benchmark
            result=benchmark(**options)
    elif command=="summarize":
        from .reporting.summary import summarize
        result=summarize(**options)
    else:
        from .reporting.plots import plot_prediction
        result=plot_prediction(args.prediction,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if command=="preflight" and not result["all_requested_checks_passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
