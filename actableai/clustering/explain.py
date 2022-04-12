from collections import defaultdict
import numpy as np


def _shorten_conditions(conds):
    re = {}
    conds = sorted(conds, key=lambda c: 0 if c[1] == ">" else 1)
    for cond in conds:
        fid, op, threshold = cond
        if op == "<=":
            re[(fid, op)] = min(re.get((fid, op), threshold + 1), threshold)
        else:
            re[(fid, op)] = max(re.get((fid, op), threshold - 1), threshold)
    return [(k[0], k[1], v) for k, v in re.items()]


def _paths_to_leaves(tree, shorten_conditions=True):
    def _walk(tree, result, node_id=0, depth=0, cond=[]):
        left_node = tree.children_left[node_id]
        right_node = tree.children_right[node_id]
        is_split_node = left_node != right_node
        if is_split_node:
            _walk(
                tree,
                result,
                left_node,
                depth + 1,
                cond + [(tree.feature[node_id], "<=", tree.threshold[node_id])],
            )
            _walk(
                tree,
                result,
                right_node,
                depth + 1,
                cond + [(tree.feature[node_id], ">", tree.threshold[node_id])],
            )
        else:
            result.append(
                {
                    "conditions": _shorten_conditions(cond)
                    if shorten_conditions
                    else cond,
                    "values": tree.value[node_id],
                    "class": np.argmax(tree.value[node_id]),
                    "precision": np.max(tree.value[node_id])
                    * 1
                    / np.sum(tree.value[node_id]),
                }
            )

    result = []
    _walk(tree, result)
    return result


def generate_cluster_descriptions(tree, features, dummy_columns=None):
    def _path_iter():
        paths = _paths_to_leaves(tree)
        for path in paths:
            description = "{:.2f}% of samples that satisfy ".format(
                path["precision"] * 100
            )
            for i, cond in enumerate(path["conditions"]):
                fid, op, threshold = cond
                if dummy_columns is not None and features[fid] in dummy_columns:
                    tokens = features[fid].split("_")
                    name = "_".join(tokens[:-1])
                    value = tokens[-1]
                    description += "{} {} {}".format(
                        name, "is" if op == ">" else "is not", value
                    )
                else:
                    description += (
                        features[fid]
                        + " "
                        + op
                        + " "
                        + (
                            "{%.2f}".format(threshold)
                            if type(threshold) is float
                            else str(threshold)
                        )
                    )
                if i == len(path["conditions"]) - 1:
                    pass
                elif i == len(path["conditions"]) - 2:
                    description += " and "
                else:
                    description += ", "
            description += " belong to this cluster"
            yield path["class"], description

    results = defaultdict(list)
    for cls, description in _path_iter():
        results[cls].append(description)
    return results
