
__name__ = "kuma"

SST = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-3,
        "lagrange_lr":     1e-2,
        "lagrange_alpha":  0.9,
    }
}

factcheck = factcheck_ood1 = factcheck_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-3,
        "lagrange_lr":     1e-2,
        "lagrange_alpha":  0.9,
    }
}


WS = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-3,
        "lagrange_lr":     1e-2,
        "lagrange_alpha":  0.9,
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
        "lasso":           0.0,
        "lambda_init":     5e-4, ## 5e-4,
        "lagrange_lr":     5e-3, ## 5e-3,
        "lagrange_alpha":  0.9,
    }
}

Yelp = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     5e-4, ## 5e-4,
        "lagrange_lr":     5e-3, ## 5e-3,
        "lagrange_alpha":  0.9,
    }
}

AmazInstr = AmazDigiMu = AmazPantry = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.0001, ##0.00001
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     5e-4,
        "lagrange_lr":     5e-3,
        "lagrange_alpha":  0.9,
    }
}

xfact = xfact_ood1 = xfact_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-3,
        "lagrange_lr":     1e-2,
        "lagrange_alpha":  0.9,
    }
}

complain = complain_ood1 = complain_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-3,
        "lagrange_lr":     1e-2,
        "lagrange_alpha":  0.9,
    }
}

bragging = bragging_ood1 = bragging_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-3,
        "lagrange_lr":     1e-2,
        "lagrange_alpha":  0.9,
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

