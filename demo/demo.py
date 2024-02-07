import time
import numpy as np

from utils import path_smoothing


# Serial Numbers of available robots. With them a fancy name to facilitate their selection
SERIAL_NUMBERS = {
    'teseo': '3JKCK160030359',
    'perseo': '3JKCK1600303HQ',
    'heracles': '3JKCK16003042N',
    'eneas': '3JKCK16003039B',
    'jason': '3JKCK7W0030DKB',
    'aquiles': '159CK9200607QK',
}


def task_with_params(actions, depot, num_actions=4, time_step=2e-2):
    import robomaster  # pip install robomaster
    def task(robot_group):

        # Get path from actions
        path = actions2path(actions, depot, num_actions, time_step)
        path = np.concatenate(
            (np.array(path)[..., 0, None], path_smoothing(np.array(path)[..., 1:].transpose(1, 0)).transpose(1, 0)),
            axis=1
        )

        # Follow the path
        position = depot
        for i, p in enumerate(path[1:]):

            # Get current node index and position
            node_idx = p[0]
            next_position = np.array(p[1:])

            # Move the robot
            dy, dx = next_position - position
            robot_group.chassis.move(dx, dy, 0, 1, 180)
            time.sleep(1)

            # Shoot when new region is visited
            if i + 2 < len(path):
                if node_idx != path[i + 2][0]:
                    robot_group.blaster.fire(fire_type=robomaster.blaster.INFRARED_FIRE, times=1)
                    time.sleep(0.1)

            # Update position
            position = next_position
    return task


def actions2path(actions, depot, num_actions=4, time_step=2e-2):
    position = depot
    path = [[0, *position]]

    # For each step
    for action in actions:

        # Get node and direction
        node_idx = action[0]
        direction = action[1]

        # Get next position
        angle = direction * 2 * np.pi / num_actions
        next_position = position + np.array([time_step * np.cos(angle), time_step * np.sin(angle)])

        # Update position and path
        position = next_position
        path.append([node_idx, *next_position])
    return path


def demo(robot_names, actions, depot, num_actions=4, time_step=2e-2):
    from multi_robomaster import multi_robot  # pip install robomaster

    # test_everything(actions, depot, num_actions, time_step)
    actions = [actions[0], actions[0]]
    depot = [depot[0]*10, depot[1]*10]
    time_step *= 5

    # Serial Numbers of available robots
    sn_list = [SERIAL_NUMBERS[r] for r in robot_names]

    # Initialize multi-robot class
    multi_robots = multi_robot.MultiEP()
    multi_robots.initialize()

    # Find robots
    number = multi_robots.number_id_by_sn(*[[i, r] for i, r in enumerate(sn_list)])
    print("The number of robot is: {0}".format(number))

    # ONE GROUP: A group containing all the robots will perform the same task simultaneously
    # group = multi_robots.build_group(range(len(sn_list)))  # The group is composed by robots: [0, 1, 2, ...]
    # multi_robots.run([group, group_task])  # Run group

    # MULTIPLE GROUPS: Each group, composed by one robot each, will perform different tasks simultaneously
    groups = [multi_robots.build_group([i]) for i in range(len(sn_list))]  # List of groups: [[0], [1], [2]...]
    multi_robots.run(                                                      # Run list of groups
        *[[group, task_with_params(
            actions[i], depot, num_actions, time_step
        )] for i, group in enumerate(groups)]
    )

    # ******************************************************************************************************************
    # *** Note that you can make combinations of groups like: [[0, 1], [2]], which means that robots 0 and 1 perform ***
    # *** the same task and robot 2 performs another different task).                                                ***
    # ******************************************************************************************************************

    # End execution
    print("Game over")
    multi_robots.close()


def test_everything(actions, depot, num_actions=4, time_step=2e-2):

    # Get path from actions
    path = actions2path(actions[0], depot, num_actions, time_step)
    # path = np.concatenate(
    #     (np.array(path)[..., 0, None], path_smoothing(np.array(path)[..., 1:].transpose(1, 0)).transpose(1, 0)), axis=1
    # )

    # Follow the path
    position = depot
    for i, p in enumerate(path[1:]):

        # Get current node index and position
        node_idx = p[0]
        next_position = np.array(p[1:])

        # Move the robot
        dy, dx = next_position - position
        print(f"Moving: {dx}, {dy} meters")

        # Shoot when new region is visited
        if i + 2 < len(path):
            if node_idx != path[i + 2][0]:
                print("SHOOT!")

        # Update position
        position = next_position
