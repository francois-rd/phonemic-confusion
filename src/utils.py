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


def int2phoneme(int_to_phoneme, int_list):
    """
    Converts a list of integers into a list of phonemes according to the given
    int-to-phoneme dictionary. If int_to_phoneme is None, then return int_list
    as is.

    :param int_to_phoneme: a dictionary mapping integers to phonemes
    :param int_list: a list of integers to map
    :return: a list of phonemes
    """
    if int_to_phoneme is None:
        return int_list
    return [int_to_phoneme.get(i, "KEY_ERROR:" + str(i)) for i in int_list]
