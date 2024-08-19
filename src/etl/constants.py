import numpy as np
import pandas as pd

MSSQL_INT_TYPES = ['bigint', 'int', 'smallint', 'tinyint']
MSSQL_FLOAT_TYPES = ['decimal', 'numeric', 'float']
MSSQL_STR_TYPES = ['varchar', 'nvarchar', 'char', 'nchar']
MSSQL_DATE_TYPES = ['date', 'datetime', 'datetime2']
NUMPY_INT_TYPES = [np.int_, np.int64, np.int32, np.int8, 'Int64']
NUMPY_FLOAT_TYPES = [np.float64, np.float32, np.float16, 'Float64']
NUMPY_STR_TYPES = [np.str_, np.object_, 'string']
NUMPY_BOOL_TYPES = [np.bool_, np.True_, np.False_, pd.BooleanDtype, 'boolean']
NUMPY_DATE_TYPES = [np.datetime64, 'datetime64[ns]']
