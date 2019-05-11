# Imports
import matplotlib.pyplot as plot



# Functions

def thresholding(s, threshold, shift):
    """This function thresholds a value according to the chosen threshold and shift

        Args:
            s (float): Value to be thresholded
            threshold (float): Thresholding factor
            shift (float): if there is a shift in the data, this can be used to compensate

        Return:


        """
    if s > (threshold + shift):
        return '1'
    elif s < (threshold * (-1)+ shift):
        return '-1'
    return '0'



# Adapted from https://www.kaggle.com/chirag19/time-series-analysis-with-python-beginner
def test_stationarity(x):
    # Determing rolling statistics
    rolmean = x.rolling(window=22, center=False).mean()

    rolstd = x.rolling(window=12, center=False).std()

    # Plot rolling statistics:
    orig = plot.plot(x, color='blue', label='Original')
    mean = plot.plot(rolmean, color='red', label='Rolling Mean')
    std = plot.plot(rolstd, color='black', label='Rolling Std')
    plot.legend(loc='best')
    plot.title('Rolling Mean & Standard Deviation')
    plot.show(block=False)

    # Perform Dickey Fuller test
    result = adfuller(x)
    print('ADF Stastistic: %f' % result[0])
    print('p-value: %f' % result[1])
    pvalue = result[1]
    for key, value in result[4].items():
        if result[0] > value:
            print("The graph is non stationery")
            break
        else:
            print("The graph is stationery")
            break;
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f ' % (key, value))


# Adapted from https://stackoverflow.com/questions/48794501/how-fast-to-derive-new-features-pandas-shift-one-period-or-n-periods-at-same-t

def df_derived_by_shift(df,lag=0,NON_DER=[]):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i*6)]
                else:
                    cols[x].append('{}_{}'.format(x, i*6))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1, join_axes=[df.index])
    return df