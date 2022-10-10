def paramSet(args: list, paramName: str) -> bool:
    """ Returns true if a parameter with the given name is contained in the given argument list. """
    return paramName in args

def argForParam(args: list, paramName: str, defaultParam = None) -> str:
    """ Returns the parameter value for the given parameter name in the given argument list. 
    If the given parameter name is not contained in the argument list, the default value is returned. """
    try:
        if paramSet(args, paramName):
            return args[args.index(paramName) + 1]
        else:
             return defaultParam
    except:
        return defaultParam

def setParam(args: list, paramName: str, paramValue = None):
    """ Sets or overrides the parameter value for the given parameter name in the given argument list. """
    if paramSet(args, paramName) and paramValue != None:
            args[args.index(paramName) + 1] = paramValue
            args[:] = args
    else:
        if paramValue == None:
            args[:] = args[:] + [paramName]
        else:
            args[:] = args[:] + [paramName, paramValue]
    return None
