from RF import RF


if __name__ == "__main__":
    rf = RF(a=1, b=100)

    nodes = [
        (-1, 1), 
        (0, 0), 
        (0.5, 0.2),
    ]

    rf.visualize(points=nodes, three_d=False)
    rf.visualize(points=nodes, three_d=True)