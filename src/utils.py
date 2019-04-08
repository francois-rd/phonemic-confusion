def discrete_value_error_message(param_name, values):
    """
    Returns "<param_name> must be one of: <values>".

    :param param_name: the name of the parameter
    :type param_name: string
    :param values: the possible values for the parameter
    :type values: iterable
    :return: "<param_name> must be one of: <values>"
    :rtype: string
    """
    values = [("'{}'".format(i) if type(i) == str else str(i)) for i in values]
    value_string = ', '.join(values[:-1]) + ', or ' + values[-1]
    return "`{}' must be one of: {}".format(param_name, value_string)