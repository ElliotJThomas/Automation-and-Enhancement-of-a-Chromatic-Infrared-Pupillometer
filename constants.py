'''
User interface constants
'''
USER_INTERFACE_NAME = "Pupillometry Image Processing"
USER_INTERFACE_INITIAL_HEIGHT = 700
USER_INTERFACE_INITIAL_WIDTH = 1000

'''
Thread constants
'''
THREAD_DELAY_PROGRAMMING_THREAD = 1
THREAD_DELAY_GUI_COMMS_THREAD = 0.1
THREAD_DELAY_TEST_THREAD = 0.001
THREAD_FLUSH_LOG_DELAY = 5

'''
Imaging constants
'''

FRAME_RATE = 16
quad_a = 1.8698*10**(-9)
quad_b = 1.0947*10**(-4)
pupil_scale = 167 #Taken from 1cm scale measured with imageJ