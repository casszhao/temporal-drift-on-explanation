
__name__ = "rl"

SST ={
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 0.
    }
}

factcheck_full = factcheck = factcheck_ood1 = factcheck_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 0.
    }
}


IMDB = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 2
    }
}

Yelp = AmazInstr = AmazDigiMu = AmazPantry = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.0001, ##0.00001
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.0,
        "coherence": 0.0
    }
}


xfact_full = xfact = xfact_ood1 = xfact_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 0.
    }
}

complain_full = complain = complain_ood1 = complain_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 0.
    }
}

bragging_full = bragging = bragging_ood1 = bragging_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 0.
    }
}

binarybragging_full = binarybragging = binarybragging_ood1 = binarybragging_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 0.
    }
}

get_ = {
    "SST": SST,
    "IMDB": IMDB,
    "Yelp" : Yelp,
    "AmazPantry" : AmazPantry,
    "AmazInstr" : AmazInstr,
    "AmazDigiMu" : AmazDigiMu,
    'factcheck_full': factcheck_full,
    'factcheck': factcheck,
    'factcheck_ood1': factcheck_ood1,
    'factcheck_ood2': factcheck_ood2,
    'xfact': xfact,
    'xfact_full': xfact_full,
    'xfact_ood1': xfact_ood1,
    'xfact_ood2': xfact_ood2,
    'complain': complain,
    'complain_full': complain_full,
    'complain_ood2': complain_ood2,
    'complain_ood1': complain_ood1,
    'bragging': bragging,
    'bragging_full': bragging_full,
    'bragging_ood1': bragging_ood1,
    'bragging_ood2': bragging_ood2,
    'binarybragging': binarybragging,
    'binarybragging_full': binarybragging_full,
    'binarybragging_ood1': binarybragging_ood1,
    'binarybragging_ood2': binarybragging_ood2,
    'AmazDigiMu': AmazDigiMu,
    'AmazDigiMu_full': AmazDigiMu_full,
    'AmazDigiMu_ood1': AmazDigiMu_ood1,
    'AmazDigiMu_ood2': AmazDigiMu_ood2,
}

