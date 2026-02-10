from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import threading

class OSCServers:
    def __init__(self):
        """
        Initialize the OSC server and motion capture data storage.
        """
        self.ip_in_supervisor = "0.0.0.0"   # inbound OSC
        self.port_in_supervisor = 5008

        self.ip_out_supervisor = "0.0.0.0"  # outbound OSC
        self.port_out_supervisor = 5007

        # Initialize the OSC server
        self.dispatcher = dispatcher.Dispatcher()
        # self.dispatcher.map("/limb_direction", self.get_GPT_direction)  # Map OSC addresses to the store_rotation function
        self.server = osc_server.BlockingOSCUDPServer((self.ip_in_supervisor, self.port_in_supervisor), self.dispatcher)
        
        # Start the server in a separate thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        print(f"OSC server running at {self.ip_in_supervisor}:{self.port_in_supervisor}")

    
    def send_message(self, message):
        client = udp_client.SimpleUDPClient(self.ip_out_supervisor, self.port_out_supervisor)
        # print("connected to GPT-main server at "+self.ip_out_supervisor+":"+str(self.port_out_supervisor))

        client.send_message("/instruction_direction", message)
    
    def send_ems_movements(self, handedness, limb, direction):
        client = udp_client.SimpleUDPClient(self.ip_out_supervisor, self.port_out_supervisor)
        # print("connected to GPT-main server at "+self.ip_out_supervisor+":"+str(self.port_out_supervisor))
        print(f"Sending OSC message: /instruction_direction {handedness} {limb} {direction}")
        client.send_message("/instruction_direction", [handedness, limb, direction])

    def send_stim_status(self, message):
        client = udp_client.SimpleUDPClient(self.ip_out_supervisor, self.port_out_supervisor)
        # print("connected to GPT-main server at "+self.ip_out_supervisor+":"+str(self.port_out_supervisor))
        if message == "stim_on":
            client.send_message("/stim_status", True)
        elif message == "stim_off":
            client.send_message("/stim_status", False)