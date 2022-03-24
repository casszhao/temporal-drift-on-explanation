# 1. bert predictive resultes -- on In domain / ood1 / ood2
# 2. different measures of different attributes rationales for both top / contigious -- on In domain / ood1 / ood2
# 3. FRESH results
# 4. kuma results
# 5. domain similarity between:  In domain / ood1 / ood2
# 6. rationale similarity between:  In domain / ood1 / ood2
# 7. datasets metadata: train/test/ size, time span, label distribution
import argparse
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "complain",
)

user_args = vars(parser.parse_args())

# 1. bert predictive resultes -- on In domain / ood1 / ood2



df =