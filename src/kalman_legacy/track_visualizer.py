import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import math
from matplotlib.widgets import Button, Slider
from loguru import logger
from matplotlib.patches import Ellipse
from matplotlib.patches import Wedge
from matplotlib.patches import Polygon as PolygonMatplot
# from shapely.geometry import Polygon
from collision import *
from collision import Vector as v


class TrackVisualizer(object):
    def __init__(self, config, tracks, static_info, meta_info, fig=None):
        self.config = config
        self.input_path = config["input_path"]
        self.recording_name = config["recording_name"]
        self.image_width = None
        self.image_height = None
        self.scale_down_factor = config["scale_down_factor"]
        self.skip_n_frames = config["skip_n_frames"]

        # Get configurations
        if self.scale_down_factor % 2 != 0:
            logger.warning("Please only use even scale down factors!")

        # Tracks information
        self.tracks = tracks
        self.static_info = static_info
        self.meta_info = meta_info
        self.maximum_frames = np.max([self.static_info[track["trackId"]]["finalFrame"] for track in self.tracks])

        # Save ids for each frame
        self.ids_for_frame = {}
        for i_frame in range(self.maximum_frames):
            indices = [i_track for i_track, track in enumerate(self.tracks)
                       if
                       self.static_info[track["trackId"]]["initialFrame"] <= i_frame <=
                       self.static_info[track["trackId"]][
                           "finalFrame"]]
            self.ids_for_frame[i_frame] = indices

        # Initialize variables
        self.current_frame = 0
        self.changed_button = False
        self.rect_map = {}
        self.plotted_objects = []
        self.track_info_figures = {}
        self.y_sign = 1

        # Create figure and axes
        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(18, 8)
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.20, top=0.99)
        else:
            self.fig = fig
            self.ax = self.fig.gca()

        self.fig.canvas.set_window_title("Recording {}".format(self.recording_name))

        # Check whether to use the given background image
        background_image_path = self.config["background_image_path"]
        if background_image_path is not None:
            self.background_image = skimage.io.imread(background_image_path)
            self.image_height = self.background_image.shape[0]
            self.image_width = self.background_image.shape[1]
            self.ax.imshow(self.background_image)
        else:
            self.background_image = np.zeros((1700, 1700, 3), dtype=np.float64)
            self.image_height = 1700
            self.image_width = 1700
            self.ax.imshow(self.background_image)

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Dictionaries for the style of the different objects that are visualized
        self.rect_style = dict(fill=True, edgecolor="k", alpha=0.4, zorder=19)
        self.triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=19)
        self.text_style = dict(picker=True, size=8, color='k', zorder=20, ha="center")
        self.text_box_style = dict(boxstyle="round,pad=0.2", alpha=.6, ec="black", lw=0.2)
        self.track_style = dict(linewidth=1, zorder=10)
        self.centroid_style = dict(fill=True, edgecolor="black", lw=0.1, alpha=1,
                                   radius=5, zorder=30)
        self.track_style_future = dict(color="linen", linewidth=1, alpha=0.7, zorder=10)
        self.circ_stylel = dict(facecolor="g", fill=True, edgecolor="r", lw=0.1, alpha=1,
                                radius=8, zorder=30)
        self.circ_styler = dict(facecolor="b", fill=True, edgecolor="r", lw=0.1, alpha=1,
                                radius=8, zorder=30)
        self.colors = dict(car="blue", truck_bus="orange", pedestrian="red", bicycle="yellow", default="green")

        # Initialize the plot with the bounding boxes of the first frame
        self.update_figure()

        ax_color = 'lightgoldenrodyellow'
        # Define axes for the widgets
        self.ax_slider = self.fig.add_axes([0.2, 0.035, 0.2, 0.04], facecolor=ax_color)  # Slider
        self.ax_button_previous2 = self.fig.add_axes([0.02, 0.035, 0.06, 0.04])
        self.ax_button_previous = self.fig.add_axes([0.09, 0.035, 0.06, 0.04])
        self.ax_button_next = self.fig.add_axes([0.44, 0.035, 0.06, 0.04])
        self.ax_button_next2 = self.fig.add_axes([0.51, 0.035, 0.06, 0.04])
        self.ax_button_play = self.fig.add_axes([0.58, 0.035, 0.06, 0.04])
        self.ax_button_stop = self.fig.add_axes([0.65, 0.035, 0.06, 0.04])

        # Define the widgets
        self.frame_slider = DiscreteSlider(self.ax_slider, 'Frame', 0, self.maximum_frames - 1,
                                           valinit=self.current_frame,
                                           valfmt='%s')
        self.button_previous2 = Button(self.ax_button_previous2, 'Previous x%d' % self.skip_n_frames)
        self.button_previous = Button(self.ax_button_previous, 'Previous')
        self.button_next = Button(self.ax_button_next, 'Next')
        self.button_next2 = Button(self.ax_button_next2, 'Next x%d' % self.skip_n_frames)
        self.button_play = Button(self.ax_button_play, 'Play')
        self.button_stop = Button(self.ax_button_stop, 'Stop')

        # Define the callbacks for the widgets' actions
        self.frame_slider.on_changed(self.update_slider)
        self.button_previous.on_clicked(self.update_button_previous)
        self.button_next.on_clicked(self.update_button_next)
        self.button_previous2.on_clicked(self.update_button_previous2)
        self.button_next2.on_clicked(self.update_button_next2)
        self.button_play.on_clicked(self.start_play)
        self.button_stop.on_clicked(self.stop_play)

        self.timer = self.fig.canvas.new_timer(interval=25 * self.skip_n_frames)
        self.timer.add_callback(self.update_button_next2, self.ax)

        # Define the callbacks for the widgets' actions
        self.fig.canvas.mpl_connect('key_press_event', self.update_keypress)

        self.ax.set_autoscale_on(False)

    def update_keypress(self, evt):
        if evt.key == "right" and self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.trigger_update()
        elif evt.key == "left" and self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.trigger_update()

    def update_slider(self, value):
        if not self.changed_button:
            self.current_frame = value
            self.trigger_update()
        self.changed_button = False

    def update_button_next(self, _):
        if self.current_frame + 1 < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning(
                "There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_next2(self, _):
        if self.current_frame + self.skip_n_frames < self.maximum_frames:
            self.current_frame = self.current_frame + self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_previous(self, _):
        if self.current_frame - 1 >= 0:
            self.current_frame = self.current_frame - 1
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def update_button_previous2(self, _):
        if self.current_frame - self.skip_n_frames >= 0:
            self.current_frame = self.current_frame - self.skip_n_frames
            self.changed_button = True
            self.trigger_update()
        else:
            logger.warning("There are no frames available with an index lower than 1.")

    def start_play(self, _):
        self.timer.start()

    def stop_play(self, _):
        self.timer.stop()

    def trigger_update(self):
        self.remove_patches()
        self.update_figure()
        self.update_pop_up_windows()
        self.frame_slider.update_val_external(self.current_frame)
        self.fig.canvas.draw_idle()

    def update_figure(self):
        # Plot the bounding boxes, their text annotations and direction arrow
        plotted_objects = []
        for track_ind in self.ids_for_frame[self.current_frame]:
            track = self.tracks[track_ind]

            track_id = track["trackId"]
            static_track_information = self.static_info[track_id]
            initial_frame = static_track_information["initialFrame"]
            current_index = self.current_frame - initial_frame

            object_class = static_track_information["class"]
            is_vehicle = object_class in ["car", "truck_bus", "motorcycle", "bicycle"]
            # In the csv file, some of the motorcycles are considered as bicycle, so here vru only include pedastrain
            is_vru = object_class in ["pedestrian"]
            bounding_box = track["bboxVis"][current_index] / self.scale_down_factor
            center_points = track["centerVis"] / self.scale_down_factor
            center_point = center_points[current_index]
            # true center point in meter
            true_center_points = track['center']
            current_velocity = np.sqrt(
                track["xVelocity"][current_index] ** 2 + track["yVelocity"][current_index] ** 2) * 3.6
            current_velocity = abs(float(current_velocity))
            current_rotation = track["heading"][current_index]
            # current_acceleration = np.sqrt(
            #     track["xAcceleration"][current_index] ** 2 + track["yAcceleration"][current_index] ** 2)

            color = self.colors[object_class] if object_class in self.colors else self.colors["default"]

            # In current frame, as moving vehicle (speed > 5km/h) point of view, print how many object should be considered
            if object_class in ["car", "truck_bus"] and current_velocity > 5:
                print("track_id", track_id)
                x0 = true_center_points[current_index][0]
                y0 = true_center_points[current_index][1]
                r = (current_velocity / 3.6) + ((current_velocity / 3.6) ** 2 / 4)
                x1 = x0 + r * math.cos(math.radians(current_rotation - 30))
                y1 = y0 + r * math.sin(math.radians(current_rotation - 30))
                x2 = x0 + r * math.cos(math.radians(current_rotation + 30))
                y2 = y0 + r * math.sin(math.radians(current_rotation + 30))
                # p1 = Polygon([(x0,y0), (x1,y1), (x2,y2)])
                p01 = Poly(v(0, 0), [v(x0, y0), v(x1, y1), v(x2, y2)])
                for othertrack_ind in self.ids_for_frame[self.current_frame]:
                    othertrack = self.tracks[othertrack_ind]
                    othertrack_id = othertrack["trackId"]
                    otherstatic_track_information = self.static_info[othertrack_id]
                    otherinitial_frame = otherstatic_track_information["initialFrame"]
                    othercurrent_index = self.current_frame - otherinitial_frame
                    othercurrent_velocity = np.sqrt(
                        othertrack["xVelocity"][othercurrent_index] ** 2 + othertrack["yVelocity"][
                            othercurrent_index] ** 2)
                    othercurrent_velocity = abs(float(othercurrent_velocity))
                    othercurrent_rotation = othertrack["heading"][othercurrent_index]
                    othertrue_center_points = othertrack['center']
                    if othertrack_ind != track_ind:
                        if object_class not in ["pedestrian"]:
                            # calculate sectors and sectors overlapping
                            # currently using only one triangle to estimate the sector
                            x4 = othertrue_center_points[othercurrent_index][0]
                            y4 = othertrue_center_points[othercurrent_index][1]
                            otherr = (othercurrent_velocity) + ((othercurrent_velocity) ** 2 / 4)
                            x5 = x4 + otherr * math.cos(math.radians(othercurrent_rotation - 30))
                            y5 = y4 + otherr * math.sin(math.radians(othercurrent_rotation - 30))
                            x6 = x4 + otherr * math.cos(math.radians(othercurrent_rotation + 30))
                            y6 = y4 + otherr * math.sin(math.radians(othercurrent_rotation + 30))
                            # p2 = Polygon([(x4,y4), (x5,y5), (x6,y6)])
                            p02 = Poly(v(0, 0), [v(x4, y4), v(x5, y5), v(x6, y6)])
                            print(collide(p01, p02), othertrack_id)
                            # TODO
                            # if collide(totalCounter++) then call the detection model, if can be detected, then detectedVehicleCounter++
                            # print(p1.intersects(p2),othertrack_id)
                        else:
                            # calculate triangle and circle overlapping
                            c02 = Circle(v(othertrue_center_points[othercurrent_index][0],
                                           othertrue_center_points[othercurrent_index][1]), othercurrent_velocity)
                            # if collide(totalCounter++) then call the detection model, if can be detected, then detectedPedestrainCounter++

                print(" ")
                # calculate the score at this timestamp for this vehicle, save it

            # Draw safety critical area
            if self.config["showSafetyCriticalArea"] and is_vru:
                # for vru should be a ellipse with radius: (VRU max speed * reaction time)
                # vru_safetycriticalarea = Ellipse((center_points[current_index][0], center_points[current_index][1]),
                #                                  current_velocity/3.6 * 0.8, current_velocity/3.6 * 1.6, angle = 270 - current_rotation)
                # Reaction time is set to 1 second
                vru_safetycriticalarea = plt.Circle((center_points[current_index][0], center_points[current_index][1]),
                                                    current_velocity / 3.6 * 1 / self.meta_info[
                                                        "orthoPxToMeter"] / self.scale_down_factor, color='g',
                                                    alpha=0.2)
                # vru_safetycriticalarea = Ellipse((center_points[current_index][0], center_points[current_index][1]), 7 * 0.8, 7 * 1.6, angle = 270 - current_rotation)
                vru_safetycriticalarea.set_facecolor('g')
                self.ax.add_patch(vru_safetycriticalarea)
                plotted_objects.append(vru_safetycriticalarea)

            if self.config["showSafetyCriticalArea"] and is_vehicle:
                # safetyCriticalAreaRadius = (current_velocity/3.6 * 1 + 0.5 * current_acceleration * (1 ** 2)) + ((current_velocity/3.6 + current_acceleration * 1)** 2 / (2 * 2))

                # Reaction time is set to 1 second and comfortable brake is set to 2
                # Based on brake model, assume during one second the vehicle is running on a constant speed as the acceleration is not always positive
                safetyCriticalAreaRadius = (current_velocity / 3.6) + ((current_velocity / 3.6) ** 2 / 4)
                # based on trajectory of after 1s second position
                # safetyCriticalAreaRadius = np.sqrt((true_center_points[current_index][0] - true_center_points[current_index + 25][0])**2 + (true_center_points[current_index][1] - true_center_points[current_index + 25][1])**2)
                # print(safetyCriticalAreaRadius)
                vehicle_safetycriticalarea = Wedge((center_points[current_index][0], center_points[current_index][1]),
                                                   safetyCriticalAreaRadius / self.meta_info[
                                                       "orthoPxToMeter"] / self.scale_down_factor / 2,
                                                   360 - current_rotation - 30, 360 - current_rotation + 30, alpha=0.2)
                vehicle_safetycriticalarea.set_facecolor('r')

                # # Draw estmiate triangle
                # if track_id == 1:
                #     drawcycle = plt.Circle((center_points[current_index][0], center_points[current_index][1]),safetyCriticalAreaRadius/self.meta_info["orthoPxToMeter"]/self.scale_down_factor,color='g',alpha=0.5)
                #     self.ax.add_patch(drawcycle)
                #     x0 = true_center_points[current_index][0]
                #     # should be nagative when y axis
                #     y0 = true_center_points[current_index][1]
                #     r = safetyCriticalAreaRadius
                #     x1 = x0 + r*math.cos(math.radians(current_rotation-30))
                #     y1 = y0 + r*math.sin(math.radians(current_rotation-30))
                #     x2 = x0 + r*math.cos(math.radians(current_rotation+30))
                #     y2 = y0 + r*math.sin(math.radians(current_rotation+30))

                #     # a = (x0/self.meta_info["orthoPxToMeter"]/self.scale_down_factor,-y0/self.meta_info["orthoPxToMeter"]/self.scale_down_factor)
                #     # b = (x1/self.meta_info["orthoPxToMeter"]/self.scale_down_factor,-y1/self.meta_info["orthoPxToMeter"]/self.scale_down_factor)
                #     # c = (x2/self.meta_info["orthoPxToMeter"]/self.scale_down_factor,-y2/self.meta_info["orthoPxToMeter"]/self.scale_down_factor)
                #     a = (x0,y0)
                #     b = (x1,y1)
                #     c = (x2,y2)
                #     vehicle_safetycriticalareaEstimate = PolygonMatplot([a,b,c], closed = True)
                #     vehicle_safetycriticalareaEstimate.set_facecolor('b')
                #     self.ax.add_patch(vehicle_safetycriticalareaEstimate)

                self.ax.add_patch(vehicle_safetycriticalarea)
                plotted_objects.append(vehicle_safetycriticalarea)

            if self.config["plotBoundingBoxes"] and is_vehicle:
                rect = plt.Polygon(bounding_box, True, facecolor=color, **self.rect_style)
                self.ax.add_patch(rect)
                plotted_objects.append(rect)

            if self.config["plotDirectionTriangle"] and is_vehicle:
                # Add triangles that display the direction of the cars
                triangle_factor = 0.75
                a_x = bounding_box[3, 0] + ((bounding_box[2, 0] - bounding_box[3, 0]) * triangle_factor)
                b_x = bounding_box[0, 0] + ((bounding_box[1, 0] - bounding_box[0, 0]) * triangle_factor)
                c_x = bounding_box[2, 0] + ((bounding_box[1, 0] - bounding_box[2, 0]) * 0.5)
                triangle_x_position = np.array([a_x, b_x, c_x])

                a_y = bounding_box[3, 1] + ((bounding_box[2, 1] - bounding_box[3, 1]) * triangle_factor)
                b_y = bounding_box[0, 1] + ((bounding_box[1, 1] - bounding_box[0, 1]) * triangle_factor)
                c_y = bounding_box[2, 1] + ((bounding_box[1, 1] - bounding_box[2, 1]) * 0.5)
                triangle_y_position = np.array([a_y, b_y, c_y])

                # Differentiate between vehicles that drive on the upper or lower lanes
                triangle_info = np.array([triangle_x_position, triangle_y_position])
                polygon = plt.Polygon(np.transpose(triangle_info), True, **self.triangle_style)
                self.ax.add_patch(polygon)
                plotted_objects.append(polygon)

            if self.config["plotTrackingLines"]:
                plotted_centroid = plt.Circle((center_points[current_index][0],
                                               center_points[current_index][1]),
                                              facecolor=color, **self.centroid_style)
                self.ax.add_patch(plotted_centroid)
                plotted_objects.append(plotted_centroid)
                if center_points.shape[0] > 0:
                    # Calculate the centroid of the vehicles by using the bounding box information
                    # Check track direction
                    plotted_centroids = self.ax.plot(
                        center_points[0:current_index + 1][:, 0],
                        center_points[0:current_index + 1][:, 1],
                        color=color, **self.track_style)
                    plotted_objects.append(plotted_centroids)
                    if self.config["plotFutureTrackingLines"]:
                        # Check track direction
                        plotted_centroids_future = self.ax.plot(
                            center_points[current_index:][:, 0],
                            center_points[current_index:][:, 1],
                            **self.track_style_future)
                        plotted_objects.append(plotted_centroids_future)

            if self.config["showTextAnnotation"]:
                # Plot the text annotation
                annotation_text = "ID{}".format(track_id)
                if self.config["showClassLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    annotation_text += "{}".format(object_class[0])
                if self.config["showVelocityLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    annotation_text += "{:.2f}km/h".format(current_velocity)
                if self.config["showRotationsLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    annotation_text += "Deg%.2f" % current_rotation
                if self.config["showAgeLabel"]:
                    if annotation_text != '':
                        annotation_text += '|'
                    age = static_track_information["age"]
                    annotation_text += "Age%d/%d" % (current_index + 1, age)
                # Differentiate between using an empty background image and using the virtual background
                target_location = (
                    center_point[0],
                    (center_point[1]))
                text_location = (
                    (center_point[0] + 45),
                    (center_point[1] - 20))
                text_patch = self.ax.annotate(annotation_text, xy=target_location, xytext=text_location,
                                              bbox={"fc": color, **self.text_box_style}, **self.text_style)
                plotted_objects.append(text_patch)

        # Add listener to figure
        self.fig.canvas.mpl_connect('pick_event', self.on_click)
        # Save the plotted objects in a list
        self.plotted_objects = plotted_objects

    def on_click(self, event):
        artist = event.artist
        text_value = artist._text
        if "ID" not in text_value:
            return

        try:
            track_id_string = text_value[:text_value.index("|")]
            track_id = int(track_id_string[2:])
            track = None
            for track in self.tracks:
                if track["trackId"] == track_id:
                    track = track
                    break
            if track is None:
                logger.error("No track with the ID {} was found. Nothing to show.".format(track_id))
                return
            static_information = self.static_info[track_id]

            # Get information of the selected track
            centroids = track["center"]
            rotations = track["heading"]
            centroids = np.transpose(centroids)
            initial_frame = static_information["initialFrame"]
            final_frame = static_information["finalFrame"]
            x_limits = [initial_frame, final_frame]
            track_frames = np.linspace(initial_frame, final_frame, centroids.shape[1], dtype=np.int64)

            # Create a new figure that pops up
            fig = plt.figure(np.random.randint(0, 5000, 1))
            fig.canvas.mpl_connect('close_event', lambda evt: self.close_track_info_figure(evt, track_id))
            fig.canvas.mpl_connect('resize_event', lambda evt: fig.tight_layout())
            fig.set_size_inches(12, 7)
            fig.canvas.set_window_title("Recording {}, Track {} ({})".format(self.recording_name,
                                                                             track_id, static_information["class"]))

            borders_list = []
            subplot_list = []

            key_check_list = ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
            counter = 3
            for check_key in key_check_list:
                if check_key in track and track[check_key] is not None:
                    counter = counter + 1
            if 3 < counter <= 6:
                subplot_index = 321
            elif 6 < counter <= 9:
                subplot_index = 331
            else:
                subplot_index = 311

            # ---------- X POSITION ----------
            sub_plot = plt.subplot(subplot_index, title="X-Position [m]")
            subplot_list.append(sub_plot)
            x_positions = centroids[0, :]
            borders = [np.amin(x_positions), np.amax(x_positions)]
            plt.plot(track_frames, x_positions)
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
            self.plotted_objects.append(red_line)
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- Y POSITION ----------
            sub_plot = plt.subplot(subplot_index, title="Y-Position [m]")
            subplot_list.append(sub_plot)
            y_positions = centroids[1, :]
            borders = [np.amin(y_positions), np.amax(y_positions)]
            plt.plot(track_frames, y_positions)
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
            self.plotted_objects.append(red_line)
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- HEADING ----------
            sub_plot = plt.subplot(subplot_index, title="Heading [deg]")
            subplot_list.append(sub_plot)
            borders = [-10, 400]
            plt.plot(track_frames, np.unwrap(rotations, discont=360))
            red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
            self.plotted_objects.append(red_line)
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "xVelocity" ----------
            if "xVelocity" in track and track["xVelocity"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="X-Velocity [m/s]")
                subplot_list.append(sub_plot)
                x_velocity = track["xVelocity"]
                borders = [np.amin(x_velocity), np.amax(x_velocity)]
                plt.plot(track_frames, x_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yVelocity" ----------
            if "yVelocity" in track and track["yVelocity"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Y-Velocity [m/s]")
                subplot_list.append(sub_plot)
                y_velocity = track["yVelocity"]
                borders = [np.amin(y_velocity), np.amax(y_velocity)]
                plt.plot(track_frames, y_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "lonVelocity" ----------
            if "lonVelocity" in track and track["lonVelocity"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Longitudinal-Velocity [m/s]")
                subplot_list.append(sub_plot)
                lon_velocity = track["lonVelocity"]
                borders = [np.amin(lon_velocity), np.amax(lon_velocity)]
                plt.plot(track_frames, lon_velocity)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "XAcceleration" ----------
            if "xAcceleration" in track and track["xAcceleration"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="X-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                x_acc = track["xAcceleration"]
                borders = [np.amin(x_acc), np.amax(x_acc)]
                plt.plot(track_frames, x_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yAcceleration" ----------
            if "yAcceleration" in track and track["yAcceleration"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Y-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                y_acc = track["yAcceleration"]
                borders = [np.amin(y_acc), np.amax(y_acc)]
                plt.plot(track_frames, y_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')
            subplot_index = subplot_index + 1

            # ---------- "yAcceleration" ----------
            if "lonAcceleration" in track and track["lonAcceleration"] is not None:
                # Plot the velocity
                sub_plot = plt.subplot(subplot_index, title="Longitudinal-Acceleration [m/s^2]")
                subplot_list.append(sub_plot)
                lon_acc = track["lonAcceleration"]
                borders = [np.amin(lon_acc), np.amax(lon_acc)]
                plt.plot(track_frames, lon_acc)
                red_line = plt.plot([self.current_frame, self.current_frame], borders, "--r")
                self.plotted_objects.append(red_line)
                borders_list.append(borders)
                plt.xlim(x_limits)
                offset = (borders[1] - borders[0]) * 0.05
                borders = [borders[0] - offset, borders[1] + offset]
                plt.ylim(borders)
                sub_plot.grid(True)
                plt.xlabel('Frame')

            self.track_info_figures[track_id] = {"main_figure": fig,
                                                 "borders": borders_list,
                                                 "subplots": subplot_list}
            plt.show()
        except Exception:
            logger.exception("An exception occured trying to display track information")
            return

    def close_track_info_figure(self, evt, track_id):
        if track_id in self.track_info_figures:
            self.track_info_figures[track_id]["main_figure"].canvas.mpl_disconnect('close_event')
            self.track_info_figures.pop(track_id)

    def get_figure(self):
        return self.fig

    def remove_patches(self):
        self.fig.canvas.mpl_disconnect('pick_event')
        for figure_object in self.plotted_objects:
            if isinstance(figure_object, list):
                figure_object[0].remove()
            else:
                figure_object.remove()
        self.plotted_objects = []

    def update_pop_up_windows(self):
        for track_id, track_map in self.track_info_figures.items():
            borders = track_map["borders"]
            subplots = track_map["subplots"]
            for subplot_index, subplot_figure in enumerate(subplots):
                new_line = subplot_figure.plot([self.current_frame, self.current_frame], borders[subplot_index], "--r")
                self.plotted_objects.append(new_line)
            track_map["main_figure"].canvas.draw_idle()

    @staticmethod
    @logger.catch(reraise=True)
    def show():
        plt.show()


class DiscreteSlider(Slider):
    def __init__(self, *args, **kwargs):
        self.inc = kwargs.pop('increment', 1)
        self.valfmt = '%s'
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        if self.val != val:
            discrete_val = int(int(val / self.inc) * self.inc)
            xy = self.poly.xy
            xy[2] = discrete_val, 1
            xy[3] = discrete_val, 0
            self.poly.xy = xy
            self.valtext.set_text(self.valfmt % discrete_val)
            if self.drawon:
                self.ax.figure.canvas.draw()
            self.val = val
            if not self.eventson:
                return
            for cid, func in self.observers.items():
                func(discrete_val)

    def update_val_external(self, val):
        self.set_val(val)
