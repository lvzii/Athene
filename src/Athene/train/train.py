def run_exp(args):
    if args.task == "ner":
        run_ner()
    elif args.task == "cls":
        run_cls()
