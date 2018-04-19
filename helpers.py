def sanitize(X):
    try:
        return X.reshape((X.shape[0], X.shape[1]))
    except:
        return X.reshape((1, len(X)))
def sanitize_data_frame(features):
    try:
        return features.values
    except:
        return features