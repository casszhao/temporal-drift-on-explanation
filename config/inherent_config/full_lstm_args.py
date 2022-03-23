__name__ = "full_lstm"

SST = IMDB = Yelp = AmazInstr = AmazDigiMu = AmazPantry = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.001, 
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "dropout":         0.1,
    }
}

complain = complain_ood1 = complain_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "dropout":         0.1,
    }
}

bragging = bragging_ood1 = bragging_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "dropout":         0.1,
    }
}

factcheck = factcheck_ood1 = factcheck_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "dropout":         0.1,
    }
}

xfact = xfact_ood1 = xfact_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "dropout":         0.1,
    }
}

get_ = {
    "SST": SST,
    "IMDB": IMDB,
    "Yelp" : Yelp,
    "AmazPantry" : AmazPantry,
    "AmazInstr" : AmazInstr,
    "AmazDigiMu" : AmazDigiMu,
    "WS": WS,
    'factcheck': factcheck,
    'factcheck_ood1': factcheck_ood1,
    'factcheck_ood2': factcheck_ood2,
    'xfact': xfact,
    'xfact_ood1': xfact_ood1,
    'xfact_ood2': xfact_ood2,
    'complain': complain,
    'complain_ood2': complain_ood2,
    'complain_ood1': complain_ood1,
    'bragging': bragging,
    'bragging_ood1': bragging_ood1,
    'bragging_ood2': bragging_ood2,
}

