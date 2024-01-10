from quasinet.qnet import load_qnet

def gentrees(modelpath,OUTDIR='trees/',format='png',threshold=1):
    model=load_qnet(modelpath,gz=False)
    model.viz_trees(tree_path=OUTDIR,
                    big_enough_threshold=threshold,format=format)
