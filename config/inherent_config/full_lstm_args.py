__name__ = "full_lstm"

SST = IMDB = Yelp = {
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

AmazDigiMu_full = AmazDigiMu_ood1 = AmazDigiMu_ood2 = AmazDigiMu = {
    "OPTIM_ARGS_": {  ## for Adam Loss
        "lr": 0.001,
        "weight_decay": 0.00001,
        "betas": [0.9, 0.999],
        "amsgrad": False,
    },
    "MODEL_ARGS_": {  ## model args
        "dropout": 0.1,
    }
}


AmazPantry_full = AmazPantry_ood1 = AmazPantry_ood2 = AmazPantry = {
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

# complain_full = complain = complain_ood1 = complain_ood2 = {
#     "OPTIM_ARGS_" : {     ## for Adam Loss
#         "lr" : 0.001, # tried 0.001
#         "weight_decay" : 0.00001,
#         "betas" : [0.9, 0.999],
#         "amsgrad" : False,
#     },
#     "MODEL_ARGS_" : {     ## model args
#         "dropout":         0.2,
#     }
# }

bragging_full = bragging = bragging_ood1 = bragging_ood2 = {
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

binarybragging_full = binarybragging = binarybragging_ood1 = binarybragging_ood2 = {
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

factcheck_full = factcheck = factcheck_ood1 = factcheck_ood2 = {
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

xfact_full = xfact = xfact_ood1 = xfact_ood2 = {
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

agnews_full = agnews = agnews_ood1 = agnews_ood2 = {
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

healthfact_full = healthfact = healthfact_ood1 = healthfact_ood2 = {
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

