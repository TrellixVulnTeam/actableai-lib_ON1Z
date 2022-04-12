class Node:
    def __init__(
        self,
        left=None,
        right=None,
        label=None,
        conditions=[],
        is_right_child=False,
        num_right_ancestors=0,
    ):
        self.left = left
        self.right = right
        self.conditions = conditions
        self.label = label
        self.is_right_child = is_right_child
        self.num_right_ancestors = num_right_ancestors


def parse_tree_dot(tree_dot):
    lines = tree_dot.split("\n")
    new_lines = []
    node_dict = {}
    for l in lines:
        if l.split(" ")[0].isnumeric():
            idx = l.find("[label=")
            if idx > 0:
                label_start = l.find("label=") + 6
                label_end = l[label_start + 1 :].find('"') + label_start
                orig_label = l[label_start + 1 : label_end + 1]
                # dropping CATE std
                idx1 = orig_label.find("\\nCATE std")
                new_label = orig_label[:idx1]
                # replacing "CATE mean" with "Mean effect (CI lower, CI upper)"
                if (new_label.find("(") > 0) and (new_label.find(")") > 0):
                    new_label = new_label.replace(
                        "CATE mean", "Average effect (CI lower, CI upper)"
                    )
                else:
                    new_label = new_label.replace("CATE mean", "Average effect")
                l = l.replace(orig_label, new_label)

                # adding node to the node_dict
                k = int(l.split(" ")[0])
                node_dict[k] = Node()
                # adding node conditions:
                conditions = []
                for l1 in orig_label.split("\\n"):
                    if (l1.find("<") > 0) or (l1.find(">") > 0):
                        conditions.append(l1)
                node_dict[k].conditions = conditions
                node_dict[k].label = new_label
            # process an edge
            else:
                parent_k = int(l.split("->")[0].strip())
                child_k = int(l.split("->")[1].strip().split(" ")[0].strip())
                if node_dict[parent_k].left is None:
                    node_dict[parent_k].left = node_dict[child_k]
                else:
                    node_dict[parent_k].right = node_dict[child_k]
                    node_dict[child_k].is_right_child = True
                if node_dict[parent_k].is_right_child:
                    node_dict[child_k].num_right_ancestors = (
                        node_dict[parent_k].num_right_ancestors + 1
                    )
        new_lines.append(l)
    return "\n".join(new_lines), node_dict


def get_cat_constraint(cat_name, cat_vals, constraints):
    """
    cat_name: name of categorical variables
    cat_vals: dictionary mapping numeric to categorical values
    constraints: list of conditions
    """
    res = []
    for v in cat_vals:
        if all([eval(c.replace(cat_name, str(v))) for c in constraints]):
            res.append(cat_vals[v])
    if len(res) == 0:
        return None
    elif len(res) == 1:
        return f"{cat_name} is {res[0]}"
    else:
        return f"{cat_name} in [" + ", ".join(["{}".format(c) for c in res]) + "]"


def negate_condition(condition):
    try:
        items = condition.split(" ")
        s = items[1]
        if s == ">=":
            ns = "<"
        elif s == "<=":
            ns = ">"
        elif s == ">":
            ns = "<="
        else:
            ns = ">="
        items[1] = ns
        return " ".join(items)
    except:
        return None


def convert_label_to_categorical(root_node, cat_name, cat_vals):
    """
    cat_name: name of the categorical variable
    cat_vals: mapping of numeric to categorical value
    """
    nodes = [root_node]
    constraints = []
    while nodes:
        cur_node = nodes.pop()
        # if a right child, negate last constraint
        if cur_node.is_right_child and len(constraints) > 0:
            last_constraint = constraints.pop()
            neg_last_constraint = negate_condition(last_constraint)
            if neg_last_constraint:
                constraints.append(neg_last_constraint)
        if cur_node.conditions:
            for c in cur_node.conditions:
                if c.find(cat_name) >= 0:
                    constraints.append(c)
            cur_constraint = get_cat_constraint(cat_name, cat_vals, constraints)
            if cur_constraint is not None:
                cur_node.label = cur_node.label.replace(
                    ",".join(cur_node.conditions), cur_constraint
                )
        if cur_node.right:
            nodes.append(cur_node.right)
        if cur_node.left:
            nodes.append(cur_node.left)
        if (
            (cur_node.right is None)
            and (cur_node.left is None)
            and (cur_node.is_right_child)
            and len(constraints) > 0
        ):
            for i in range(cur_node.num_right_ancestors + 1):
                constraints.pop()


def make_pretty_tree(tree_dot, cat_names, cat_vals_list):
    new_tree_dot, node_dict = parse_tree_dot(tree_dot)
    root_node = node_dict[0]
    for cat_name, cat_vals in zip(cat_names, cat_vals_list):
        convert_label_to_categorical(root_node, cat_name, cat_vals)
    lines = new_tree_dot.split("\n")
    new_lines = []
    for l in lines:
        if l.split(" ")[0].isnumeric():
            idx = l.find("[label=")
            if idx > 0:
                k = int(l.split(" ")[0])
                label_start = l.find("label=") + 6
                label_end = l[label_start + 1 :].find('"') + label_start
                orig_label = l[label_start + 1 : label_end + 1]
                new_label = node_dict[k].label
                l = l.replace(orig_label, new_label)
        new_lines.append(l)
    final_tree_dot = "\n".join(new_lines)
    return final_tree_dot
