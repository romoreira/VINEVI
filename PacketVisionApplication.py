class PacketVisionApplication(object):
    def main(self):
        packetVisionService = PacketVisionService()
        packetVisionService = MonitoringPacketVisionService(packetVisionService)
        

if __name__ == "__main__":
    packetVisionApplication = PacketVisionApplication()
    packetVisionApplication.main()