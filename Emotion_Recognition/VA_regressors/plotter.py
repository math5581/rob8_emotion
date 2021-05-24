import cv2
import numpy as np

# Plot values in opencv program
class Plotter:
    def __init__(self,dim):
        self.width = dim[0]
        self.height = dim[1]
        self.color = (255, 0, 0)
        self.val = []
        self.plot_canvas = np.ones((self.height, self.width, 3)) * 255

    def plot(self, val, label="plot"):
        valence=(val[0]+1)/2*self.width
        arousal=self.width-(val[1]+1)/2*self.width
        self.valence=val[0]
        self.arousal=val[1]
        self.val.append((int(valence),int(arousal)))
        return self.show_plot(label)

    def show_plot(self, label):
        self.plot_canvas = np.ones((self.height, self.width, 3)) * 255
        cv2.line(self.plot_canvas, (0, int(self.height / 2)), (self.width, int(self.height / 2)), (0, 0, 0), 1)
        cv2.line(self.plot_canvas, (int(self.width / 2), 0), (int(self.height / 2), self.height), (0, 0, 0), 1)
        for i in range(0,len(self.val)):
            #print((self.val[i][0],self.val[i][1]))
            cv2.circle(self.plot_canvas, (self.val[i][0],self.val[i][1]),3, self.color, 5)
            cv2.putText(self.plot_canvas,"valence "+str(self.valence),(self.val[i][0]+5,self.val[i][1]-10),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 1)
            cv2.putText(self.plot_canvas,"arousal "+str(self.arousal),(self.val[i][0]+5,self.val[i][1]+10),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 1)
            self.val.pop()
        return self.plot_canvas
        #cv2.imshow(label, self.plot_canvas)
        #cv2.waitKey(10)

## Test on sample files
