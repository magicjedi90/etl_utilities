import numpy as np
import pandas as pd

DB_INT_TYPES = ['bigint', 'int', 'smallint', 'tinyint', 'mediumint', 'integer']
DB_FLOAT_TYPES = ['decimal', 'numeric', 'float', 'double']
DB_STR_TYPES = ['varchar', 'nvarchar', 'char', 'nchar', 'string']
DB_BOOL_TYPES = ['bit', 'bool', 'boolean']
DB_DATE_TYPES = ['date', 'datetime', 'datetime2', 'time', 'timestamp', 'smalldatetime', 'year']
DB_TYPES = [DB_INT_TYPES, DB_FLOAT_TYPES, DB_STR_TYPES, DB_BOOL_TYPES, DB_DATE_TYPES]
NUMPY_INT_TYPES = [np.int_, np.int64, np.int32, np.int8, 'Int64']
NUMPY_FLOAT_TYPES = [np.float64, np.float32, np.float16, 'Float64']
NUMPY_STR_TYPES = [np.str_, np.object_, 'string']
NUMPY_BOOL_TYPES = [np.bool_, np.True_, np.False_, pd.BooleanDtype, 'boolean']
NUMPY_DATE_TYPES = [np.datetime64, 'datetime64[ns]']
