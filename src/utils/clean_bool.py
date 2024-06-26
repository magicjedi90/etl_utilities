def clean_bool(dirty_bool):
    if dirty_bool is None:
        return
    dirty_bool = str(dirty_bool).lower()
    if dirty_bool in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif dirty_bool in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (dirty_bool,))
