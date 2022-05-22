
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

yelp = yelp_full = yelp_ood1 = yelp_ood2 = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     1e-5, ## 5e-4, # how short or long rationales are # compare to lstm standard,
                                 # penalise when a rationale is long. So if they high they will select less text and therefore closer accuracy to only lstm
                                #so if want to increase rationale len, put lower num to let it select more text         
        "lagrange_lr":     5e-3, ## 5e-3,
        "lagrange_alpha":  0.9,
    }
}



AmazDigiMu = AmazDigiMu_full = AmazDigiMu_ood1 = AmazDigiMu_ood2 = {
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

AmazPantry = AmazPantry_full = AmazPantry_ood1 = AmazPantry_ood2 = {
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

agnews = agnews_full = agnews_ood1 = agnews_ood2 = {
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

healthfact = healthfact_full = healthfact_ood1 = healthfact_ood2 = {
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
        "lr" : 1e-5, # || MAYBE decrease# can try lower, tried 0.00001
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "lasso":           0.0,
        "lambda_init":     0.1, # how short or long rationales are # compare to lstm standard,
                                 # penalise when a rationale is long. So if they high they will select less text and therefore closer accuracy to only lstm
        "lagrange_lr":     5e-4, # have tried 1e-2, 1e-3
        "lagrange_alpha":  0.8, # have tried 0.9, 0.8
    }
}

# complain_full = complain = complain_ood1 = complain_ood2 = {
#     "OPTIM_ARGS_" : {     ## for Adam Loss
#         "lr" : 1e-5, # || MAYBE decrease# can try lower, tried 0.00001
#         "weight_decay" : 0.00001,
#         "betas" : [0.9, 0.999],
#         "amsgrad" : False,
#     },
#     "MODEL_ARGS_" : {     ## model args
#         "lasso":           0.0,
#         "lambda_init":     0.5, # || 1e-4
#         "lagrange_lr":     1e-5, # have tried 1e-2, 1e-3 || 
#         "lagrange_alpha":  0.8, # have tried 0.9, 0.8, 0.7 || 0.85 penalty to rationales length
#     }
# }



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

binarybragging_full = binarybragging = binarybragging_ood1 = binarybragging_ood2 = {
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
    "yelp": yelp,
    "yelp_full": yelp_full,
    "yelp_ood1": yelp_ood1,
    "yelp_ood2": yelp_ood2,
    'factcheck_full': factcheck_full,
    'factcheck': factcheck,
    'factcheck_ood1': factcheck_ood1,
    'factcheck_ood2': factcheck_ood2,
    'xfact': xfact,
    'xfact_full': xfact_full,
    'xfact_ood1': xfact_ood1,
    'xfact_ood2': xfact_ood2,
    # 'complain': complain,
    # 'complain_full': complain_full,
    # 'complain_ood2': complain_ood2,
    # 'complain_ood1': complain_ood1,
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
    "AmazPantry": AmazPantry,
    "AmazPantry_full": AmazPantry_full,
    "AmazPantry_ood1": AmazPantry_ood1,
    "AmazPantry_ood2": AmazPantry_ood2,
    "agnews": agnews,
    "agnews_full": agnews_full,
    "agnews_ood1": agnews_ood1,
    "agnews_ood2": agnews_ood2,
    "healthfact": healthfact,
    "healthfact_full": healthfact_full,
    "healthfact_ood1": healthfact_ood1,
    "healthfact_ood2": healthfact_ood2,
}

