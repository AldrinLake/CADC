def get_model(model_name, args):
    name = model_name.lower()
    if name == "cadc":
        from models.CADC import CADC
        return CADC(args)
    elif name == "lwf":
        from models.lwf import LwF
        return LwF(args)
    elif name == "finetune":
        from models.finetune import Finetune
        return Finetune(args)


    else:
        assert 0
