import cv2
import matplotlib.pyplot as plt
import open3d as o3d

def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def bgr2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def show_img(img, title="image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def visualize_PCD(points, colors):
#     """
#     Visualize a point cloud using matplotlib
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
#     plt.show()

def visualize_PCD(points, colors):
    """
    visualize a point cloud using open3d
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    