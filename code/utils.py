def link_list(list_array: list[list]):
    new_list = []
    for ll in list_array:
        for v in ll:
            new_list.append(v)
    return new_list


class node:
    def __init__(self, num, left, right, is_leaf, name) -> None:
        self.num = num
        self.left = left
        self.right = right
        self.name = name
        self.is_leaf = is_leaf

    def print_node(self):
        print(
            "number:{}, left:{}, right:{}, name:{}, is_leaf:{}".format(
                self.num, self.left, self.right, self.name, self.is_leaf
            )
        )


def get_sub_str(string: str, start: int, end_point: str):
    end = start
    while string[end] != end_point:
        end += 1
    return string[start:end]


def deduplicate(a: list):
    res = []
    for i in a:
        if i not in res:
            res.append(i)
    return res


def extract_tree_from_dot(ori_data: str):
    nodes = {}

    def get_node(line: str):
        if line.find("->") > 0:
            num1 = int(
                get_sub_str(
                    line,
                    0,
                    " ",
                )
            )
            num2 = int(
                get_sub_str(
                    line,
                    line.find("-> ") + 3,
                    " ",
                )
            )
            parent: node = nodes[num1]
            if parent.left == -1:
                parent.left = num2
            else:
                parent.right = num2
            return -1, None

        # print(line)
        node_number = get_sub_str(
            line,
            0,
            " ",
        )

        word = get_sub_str(
            line,
            line.find('label="') + 7,
            " ",
        )
        class_name = get_sub_str(
            line,
            line.find("class = ") + 8,
            '"',
        )

        if word in ("gini", "entropy", "log_loss"):
            return int(node_number), node(node_number, -1, -1, True, class_name)
        else:
            return int(node_number), node(node_number, -1, -1, False, word)

    ori_data = ori_data.split("\n")
    start_line = 0
    while ori_data[start_line][0] != "0":
        start_line += 1
        if start_line == len(ori_data):
            print("no start in the dot str")
            return
    for i in range(start_line, len(ori_data) - 1):
        description = ori_data[i]
        node_num, tmp_node = get_node(description)
        nodes[node_num] = tmp_node

    tree_path_list = []

    def deep_first_search(number, tree_path):
        tree_path += nodes[number].name + ","
        if nodes[number].is_leaf:
            tree_path_list.append(tree_path)
            return
        if nodes[number].left != -1:
            deep_first_search(nodes[number].left, tree_path)
        if nodes[number].right != -1:
            deep_first_search(nodes[number].right, tree_path)
        # if node_num == -1:
        #     continue
        # if tmp_node.is_leaf:
        #     print(description)
        #     tmp_node.print_node()
        #     break

    deep_first_search(0, "")
    # print(tree_path_list)
    return tree_path_list


a = "0,"
print(a.split(","))
