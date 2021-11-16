//
// Created by rodrigo on 01/09/2021.
// To compile: gcc watcher.c -lpcap

#include <stdio.h>
#include <time.h>
#include <pcap.h>
#include <netinet/in.h>
#include <netinet/if_ether.h>

void my_packet_handler(
    u_char*                     args,
    const struct pcap_pkthdr*   header,
    const u_char*               packet
);

void print_packet_info(const u_char* packet, struct pcap_pkthdr packet_header);

int main(int argc, char *argv[]) {
    char* device;
    char error_buffer[PCAP_ERRBUF_SIZE];
    pcap_t* handle;
    int timeout_limit = 5000; /* In milliseconds */

    device = argv[1];
    //printf("Device to be Sniffed: %s\n", device);
    if (device == NULL) {
        printf("Error finding device. Check parameter passing: %s\n", error_buffer);
        return 1;
    }

    /* Open device for live capture */
    handle = pcap_open_live(
        device,
        BUFSIZ,
        0,
        timeout_limit,
        error_buffer
    );

    if (handle == NULL) {
        fprintf(stderr, "Could not open device %s: %s\n", device, error_buffer);
        return 2;
    }

    pcap_loop(handle, 0, my_packet_handler, NULL);

    return 0;
}

void my_packet_handler(u_char* args, const struct pcap_pkthdr* packet_header, const u_char* packet_body){
    print_packet_info(packet_body, *packet_header);
    return;
}

void print_packet_info(const u_char* packet, struct pcap_pkthdr packet_header) {
    for (int i=0; i<packet_header.len; i++) {
        fprintf(stdout, "%02hhX ", packet[i]);
    }
    printf("\n");
}
