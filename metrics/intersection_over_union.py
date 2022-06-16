from tensorflow.keras import backend


def iou(y_true, y_pred, label=1):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = backend.cast(backend.equal(backend.argmax(y_true), label), backend.floatx())
    y_pred = backend.cast(backend.equal(backend.argmax(y_pred), label), backend.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = backend.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = backend.sum(y_true) + backend.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return backend.switch(backend.equal(union, 0), 1.0, intersection / union)

def iou_1(y_true, y_pred, label=1):
    return iou(y_true, y_pred, label=label)

def iou_2(y_true, y_pred, label=2):
    return iou(y_true, y_pred, label=label)

def iou_3(y_true, y_pred, label=3):
    return iou(y_true, y_pred, label=label)

def iou_4(y_true, y_pred, label=4):
    return iou(y_true, y_pred, label=label)

def iou_5(y_true, y_pred, label=5):
    return iou(y_true, y_pred, label=label)

def iou_6(y_true, y_pred, label=6):
    return iou(y_true, y_pred, label=label)

def iou_7(y_true, y_pred, label=7):
    return iou(y_true, y_pred, label=label)

def iou_8(y_true, y_pred, label=8):
    return iou(y_true, y_pred, label=label)

def iou_9(y_true, y_pred, label=9):
    return iou(y_true, y_pred, label=label)

def iou_10(y_true, y_pred, label=10):
    return iou(y_true, y_pred, label=label)

def iou_11(y_true, y_pred, label=11):
    return iou(y_true, y_pred, label=label)

def iou_12(y_true, y_pred, label=12):
    return iou(y_true, y_pred, label=label)

def iou_13(y_true, y_pred, label=13):
    return iou(y_true, y_pred, label=label)

def iou_14(y_true, y_pred, label=14):
    return iou(y_true, y_pred, label=label)

def iou_15(y_true, y_pred, label=15):
    return iou(y_true, y_pred, label=label)

def iou_16(y_true, y_pred, label=16):
    return iou(y_true, y_pred, label=label)

def iou_17(y_true, y_pred, label=17):
    return iou(y_true, y_pred, label=label)

def iou_18(y_true, y_pred, label=18):
    return iou(y_true, y_pred, label=label)

def iou_19(y_true, y_pred, label=19):
    return iou(y_true, y_pred, label=label)

def iou_20(y_true, y_pred, label=20):
    return iou(y_true, y_pred, label=label)

def iou_21(y_true, y_pred, label=21):
    return iou(y_true, y_pred, label=label)

def iou_22(y_true, y_pred, label=22):
    return iou(y_true, y_pred, label=label)

def iou_23(y_true, y_pred, label=23):
    return iou(y_true, y_pred, label=label)

def iou_24(y_true, y_pred, label=24):
    return iou(y_true, y_pred, label=label)

def iou_25(y_true, y_pred, label=25):
    return iou(y_true, y_pred, label=label)

def iou_26(y_true, y_pred, label=26):
    return iou(y_true, y_pred, label=label)

def iou_27(y_true, y_pred, label=27):
    return iou(y_true, y_pred, label=label)

def iou_28(y_true, y_pred, label=28):
    return iou(y_true, y_pred, label=label)

def iou_29(y_true, y_pred, label=29):
    return iou(y_true, y_pred, label=label)

def iou_30(y_true, y_pred, label=30):
    return iou(y_true, y_pred, label=label)
               
def iou_31(y_true, y_pred, label=31):
    return iou(y_true, y_pred, label=label)

def iou_32(y_true, y_pred, label=32):
    return iou(y_true, y_pred, label=label)

def iou_33(y_true, y_pred, label=33):
    return iou(y_true, y_pred, label=label)

def iou_34(y_true, y_pred, label=34):
    return iou(y_true, y_pred, label=label)

def iou_35(y_true, y_pred, label=35):
    return iou(y_true, y_pred, label=label)

def iou_36(y_true, y_pred, label=36):
    return iou(y_true, y_pred, label=label)

def iou_37(y_true, y_pred, label=37):
    return iou(y_true, y_pred, label=label)

def iou_38(y_true, y_pred, label=38):
    return iou(y_true, y_pred, label=label)

def iou_39(y_true, y_pred, label=39):
    return iou(y_true, y_pred, label=label)

def iou_40(y_true, y_pred, label=40):
    return iou(y_true, y_pred, label=label)

def iou_41(y_true, y_pred, label=41):
    return iou(y_true, y_pred, label=label)

def iou_42(y_true, y_pred, label=42):
    return iou(y_true, y_pred, label=label)

def iou_43(y_true, y_pred, label=43):
    return iou(y_true, y_pred, label=label)
