import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.misc import imread

class AnimationClass:
    def __init__(self, animation_interval, x, x_arr, gamma, runtime_seconds=10, delta_t=0.005,
                 gif_name="test.gif", speed=1):
        self.fig = plt.figure(figsize=(8, 6), dpi=200)
        self.ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
        self.l1, = plt.plot([], [], color='b')
        self.l2, = plt.plot([], [], color='g')
        self.x = x
        self.gamma = gamma
        self.gif_name = gif_name
        self.anim = None
        self.animation_interval = animation_interval
        self.animation_speed = speed
        self.velocity = 1

        # Append to x_arr such that we have the boundary conditions
        self.x_arr = x_arr
        self.include_boundary_conditions()

        # self.time_step = int(len(self.x_arr)/(self.fps*self.runtime_seconds))
        # TODO: Clean up all this mess regarding frames and playback speed
        self.n_frames = int(runtime_seconds/delta_t/animation_interval)

        #We want to add vertical lines to the plot wherever diff_gamma is non-zero
        self.gamma_change_highlighter()

    def include_boundary_conditions(self):
        # TODO Add ability to have boundary condition other than just zero
        x_arr_temp = list(map(lambda x: np.append(x, 0), self.x_arr))
        self.x_arr = list(map(lambda x: np.insert(x, 0, 0), x_arr_temp))
        return self.x_arr

    def animate(self, i):
        self.l1.set_data(self.x, np.real(self.x_arr[i]))
        self.l2.set_data(self.x, np.imag(self.x_arr[i]))
        return [self.l1, self.l2]

    def animate_with_velocity(self, i):
        # We want to get the self.x_arr data and create a version which shifts over time by velocity at each timestep
        # First to do this we must get self.x and extend it. We know that the spacing in self.x is linear so we can find
        # the spacing by doing (last-first)/num_samples
        spacing = (self.x[-1]-self.x[0])/self.x.shape[0]
        # We update this by using an array of size   self.x.shape[0] + number_of_time_steps*velocity
        number_of_elements = self.x.shape[0] + len(self.x_arr)*self.velocity
        extended_x = np.linspace(start=self.x[0], stop=self.x[-1] + spacing * number_of_elements ,num= number_of_elements)
        # We want to get x_arr and append time_step * velocity  zeros to the beginning of each sample
        new_x_arr = np.zeros(number_of_elements, dtype=np.complex128)
        new_x_arr[i*self.velocity: i*self.velocity + len(self.x_arr[i])] = self.x_arr[i]
        # We want to update self.ax such that it now has new x-limits
        self.ax.set_xlim((0, spacing * number_of_elements))
        self.l1.set_data(extended_x, np.real(new_x_arr))
        self.l2.set_data(extended_x, np.imag(new_x_arr))
        return [self.l1, self.l2]

    def initialize(self):
        self.l1.set_data([], [])
        self.l2.set_data([], [])

    def run_animation(self):
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.initialize,
                                            frames=self.n_frames, interval=self.animation_speed*self.animation_interval)
        plt.show()

    def run_animation_with_propagation(self):
        self.anim = animation.FuncAnimation(self.fig, self.animate_with_velocity, init_func=self.initialize,
                                            frames=self.n_frames, interval=self.animation_speed*self.animation_interval)
        spacing = (abs(self.x[-1]-self.x[0]))/self.x.shape[0]
        number_of_elements = self.x.shape[0] + len(self.x_arr)*self.velocity
        right_side_boundary = self.x[-1] + spacing * number_of_elements
        plt.imshow(imread("fiber_optic.png"), zorder=0, extent=[0, right_side_boundary, -1, 1],aspect='auto')
        plt.show()

    def gamma_change_highlighter(self):
        spacing = (self.x[-1]-self.x[0])/self.x.shape[0]
        number_of_elements = self.x.shape[0] + len(self.x_arr)*self.velocity
        size = self.x[0] + self.x[-1] + spacing * number_of_elements
        diff_gamma = np.diff(self.gamma)
        blue = np.argwhere(diff_gamma > 0)
        red  = np.argwhere(diff_gamma < 0)
        blue = blue/self.gamma.size
        red = red/self.gamma.size
        for b in blue:
            plt.axvline(x=b*size, color='blue')

        for r in red:
            plt.axvline(x=r*size, color='red')


    def create_gif(self):
        create_gif_input = input("Do you want to create a gif of the animation? (y/n)")
        if create_gif_input == "n":
            return
        print("Creating gif")
        if self.anim is None:
            print("Please run animate in order to create the animation object for gif creation")
            return
        else:
            self.anim.save("multimedia/" + str(self.gif_name) + ".gif", dpi=200, writer='ffmpeg', codec="libx265")
        print("Gif created with name: ", self.gif_name)

    def create_video(self):
        create_video_input = input("Do you want to create a video of the animation? (y/n)")
        if create_video_input == "n":
            return
        print("Creating video")
        if self.anim is None:
            print("Please run animate in order to create the animation object for video creation")
            return
        else:
            # TODO: Fix fps of the exported video to match that of the animation
            self.anim.save("multimedia/" + str(self.gif_name) + ".mp4", dpi=200, writer="ffmpeg", codec="libx265")
        print("Video created with name: ", self.gif_name)
