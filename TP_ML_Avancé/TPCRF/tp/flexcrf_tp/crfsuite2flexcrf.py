# Author: Slim ESSID <slim.essid@telecom-paristech.fr>
# License: LGPL v3

import numpy as np
import copy as cp

from flexcrf_tp.feature_extraction.linear_chain import LinearChainData, FeatFunIndex

# -- Helper functions----------------------------------------------------------

def add_new_features(f_xy_desc, features, y):
    """
    To deal with new features, i.e. unobserved during training: add them in
    f_xy_desc_, t_xyy_desc_ etc.
    """
    for feat in features:
        if feat not in f_xy_desc:
                f_xy_desc[feat] = {y: y}
        elif y not in f_xy_desc[feat]:
                f_xy_desc[feat][y] = y
    return f_xy_desc


def add_new_transitions(t_xyy_desc, y1, y2):
    if (y1, y2) not in t_xyy_desc['label_tr']:
        t_xyy_desc['label_tr'][(y1, y2)] = (y1, y2)
    return t_xyy_desc


def update_model_structure(feat_seq, y, f_xy_desc, t_xyy_desc):
    """
    Deal with new features, i.e. unobserved during training: add them in
    f_xy_desc, t_xyy_desc_ and return the latter.

    """

    for i, (y1, y2) in enumerate(zip(y[:-1], y[1:])):
        add_new_transitions(t_xyy_desc, y1, y2)
        if i == 0:
            add_new_features(f_xy_desc, feat_seq[i], y1)
        add_new_features(f_xy_desc, feat_seq[i], y2)

    # last token in sequence is yet to be added (not processed above):
    add_new_features(f_xy_desc, feat_seq[-1], y[-1])

    g_xy_desc = []
    for feat in f_xy_desc:
        g_xy_desc.append(('o1', feat, 1, set(f_xy_desc[feat].keys())))

    return f_xy_desc, g_xy_desc, t_xyy_desc


def set_indicator_feature_values(X, f_xy, ND):
    """
    Set values if indicator features for sequence of attributes in X.
    When features are just indicators, values of f_xy = 1
    for all attributes observed and 0 otherwise.
    """

    i = -1  # this is for when len(X)<2 and the next loop is skipped
    for i, tok_attributes in enumerate(X):
        f_xy.set(1, i, y1=ND, feat=tok_attributes)

    # set values of all observed transitions to 1
    f_xy.set(1, slice(1, f_xy.n_obs), y1=f_xy.y1_values())
    #  hence merely ignoring y1=ND (not part of f_xy.y1_values()) and starting
    # at t=1 (not t=0)

    return f_xy


def copy_crfsuite_model_params(model, flex_index):
    """ Copy theta coefs """

    label_names = {int(v): k for (k, v) in model.labels.items()}

    theta = np.zeros(flex_index.n_feat_fun)
    for arg in flex_index.index:
        y1, y2, attr, feat_ind = arg[0], arg[1], arg[6], arg[8]
        if y1 == flex_index.ND:
            if (attr, label_names[y2]) in model.state_features:
                theta[feat_ind] = \
                    model.state_features[attr, label_names[y2]]
        elif (label_names[y1], label_names[y2]) in model.transitions:
            theta[feat_ind] = \
                model.transitions[label_names[y1], label_names[y2]]
    return theta


def convert_data_to_flexcrf(data, model, n_seq=None):
    """ Copy theta coefs and convert data to flexcrf format based on
        attributes defined in model and new observed on test sentences stored
        in data. Process the first n_seq sequences.
    """

    print("\nconverting to flexcrf format...")
    n_labels = len(model.labels)
    label_set = set(range(n_labels))

    # -- Observation feature functions

    # type o1: f(x,y=l) = g(x,y=l)*I(y=l): different theta for each l
    feats = set([k[0] for k in model.state_features])
    f_xy_desc = {k: {} for k in feats}
    for (feat, y), coef in model.state_features.items():
        y = int(model.labels[y])
        f_xy_desc[feat][y] = y
    print("f_xy_desc created.")

    # -- Transition feature functions

    # type t1: I(y_{i-1}=l, y_i=q)
    t_xyy_desc = {'label_tr': {}}
    for (y1, y2), coef in model.transitions.items():
        y1, y2 = int(model.labels[y1]), int(model.labels[y2])
        t_xyy_desc['label_tr'][(y1, y2)] = (y1, y2)
    print("t_xyy_desc created")

    h_xyy_desc = [('t1', 'label_tr', 1, {})]
    # (<index>, <type>, <tag>, <dimension>, <labels> )
    # {} to indicate t1 feature, no h(x,y)

    # -- Process sentences ----------------------------------------------------
    ND = -1
    dataset = []
    thetas = []
    if n_seq is None:
        n_seq = len(data['y'])
    for seq, (X, y) in enumerate(zip(data['X'][:n_seq], data['y'][:n_seq])):
        print('Processing sentence %d/%d...' % (seq+1, n_seq))

        f_xy_desc_ = cp.copy(f_xy_desc)
        t_xyy_desc_ = cp.copy(t_xyy_desc)

        n_obs = len(y)
        y = np.array([int(model.labels[s]) for s in y])

        # deal with new features, i.e. unobserved during training: add them
        f_xy_desc_, g_xy_desc_, t_xyy_desc_ = \
            update_model_structure(X, y, f_xy_desc, t_xyy_desc)
        # create FeatFunIndex
        flex_index = FeatFunIndex(g_xy_desc_, h_xyy_desc, label_set,
                                  f_xy_desc=f_xy_desc_, t_xyy_desc=t_xyy_desc_,
                                  ND=ND)
        # create flexcrf f_xy LinearChainData;
        f_xy = LinearChainData(flex_index.index,
                               data=np.zeros((n_obs, flex_index.n_feat)))
        # fillin feature values
        f_xy = set_indicator_feature_values(X, f_xy, ND)

        dataset += [(f_xy, y)]

        theta = copy_crfsuite_model_params(model, flex_index)
        thetas.append(theta)

    return dataset, thetas

