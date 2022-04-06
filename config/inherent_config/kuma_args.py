
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

factcheck_full = factcheck = factcheck_ood1 = factcheck_ood2 = {
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

xfact_full = xfact = xfact_ood1 = xfact_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-3, # how short or long rationales are # compare to lstm standard
        "lagrange_lr":     1e-2, #
        "lagrange_alpha":  0.9,
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
        "lasso":           0.0,
        "lambda_init":     1e-3,
        "lagrange_lr":     1e-2,
        "lagrange_alpha":  0.9,
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
}

