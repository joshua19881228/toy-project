#ifdef _DP
#ifndef _DEBUG
#include "api/license++.h"
#include "pc-identifiers.h"
#endif
#endif

#include "./util.h"
#include <fstream>
#include <iostream>
#include <strstream>

using std::ifstream;
using std::strstream;

#include <stdio.h> 
#include <time.h>

#ifdef _DP
#ifndef _DEBUG
#pragma comment(lib,"license++_static.lib")
#pragma comment(lib,"os.lib")
#pragma comment(lib,"base.lib")
#pragma comment(lib,"tools_base.lib")
#endif
#endif

bool DLLEXPORT verifyLicense(const string& license_path)
{
#ifdef _DP
#ifndef _DEBUG
    LicenseInfo licenseInfo;
    LicenseLocation licenseLocation;
    licenseLocation.openFileNearModule = false;
    licenseLocation.licenseFileLocation = license_path.c_str();
    licenseLocation.environmentVariableName = "";

    EVENT_TYPE result = acquire_license("dpcnn", licenseLocation, &licenseInfo);
    if (result != LICENSE_OK) {
        return false;
    }
    else
        return true;
#endif
#endif
    return true;
}

int loadConfigFile(const string& config_file_path, map<string, string>& configs)
{
#ifndef _DEBUG
    //const string licLocation("license.lic");
    //LicenseInfo licenseInfo;
    //LicenseLocation licenseLocation;
    //licenseLocation.openFileNearModule = false;
    //licenseLocation.licenseFileLocation = licLocation.c_str();
    //licenseLocation.environmentVariableName = "";

    //EVENT_TYPE result = acquire_license("dpcnn", licenseLocation, &licenseInfo);
    //if (result != LICENSE_OK){
    //    exit(0);
    //}
#endif
    ifstream in_file(config_file_path.c_str(), std::ios::in);
    if (!in_file)
    {
        LOGW("load config file %s failed\n", config_file_path.c_str());
        return -1;
    }

    while (!in_file.eof())
    {
        std::string a_line, key, value;
        getline(in_file, a_line);
        if (a_line.length() < 2)
            continue;
        if (a_line.find(" ") != a_line.rfind(" "))
        {
            LOGW("config file %s has wrong format\n", config_file_path.c_str());
            return -2;
        }
        strstream a_stream;
        a_stream << a_line;
        a_stream >> key >> value;
        configs[key] = value;
    }
    in_file.close();
    return 0;
}


#ifdef __ANDROID__
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <sys/ioctl.h>
const int ADDR_lIST_LENGTH = 3;
string addr_list[ADDR_lIST_LENGTH] = { "c0:21:0d:4d:6c:bd", "cc:79:cf:bc:ee:e9", "00:00:ff:ff:00:00" };
bool getMac()
{
#ifdef USE_INNER_TZTX
    return true;
#endif
    char mac_addr[18] = { 0 };
    //GetMACAddress
    struct ifreq tmp;
    int sock_mac;
    sock_mac = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_mac == -1)
    {
        //LOGE("Create socket error!");
        return false;
    }
    memset(&tmp, 0, sizeof(tmp));
    strncpy(tmp.ifr_name, "wlan0", sizeof(tmp.ifr_name) - 1);//eth0 pc
    if ((ioctl(sock_mac, SIOCGIFHWADDR, &tmp)) < 0)
    {
        //LOGE("ioctl error!");
        return false;
    }
    sprintf(mac_addr, "%02x:%02x:%02x:%02x:%02x:%02x",
        (unsigned char)tmp.ifr_hwaddr.sa_data[0],
        (unsigned char)tmp.ifr_hwaddr.sa_data[1],
        (unsigned char)tmp.ifr_hwaddr.sa_data[2],
        (unsigned char)tmp.ifr_hwaddr.sa_data[3],
        (unsigned char)tmp.ifr_hwaddr.sa_data[4],
        (unsigned char)tmp.ifr_hwaddr.sa_data[5]
    );
    close(sock_mac);
    //LOGE("MAC:%s", mac_addr);
    string mac_str = mac_addr;
    for (int i = 0; i < ADDR_lIST_LENGTH; i++) {
        if (mac_str.compare(addr_list[i]) == 0)
            return true;
    }
    return false;
}
#endif

#ifdef __linux__
#ifndef __ANDROID__
#include <sys/ioctl.h>
#include <net/if.h> 
#include <unistd.h>
#include <netinet/in.h>
#include <string.h>
bool getMacAddress(unsigned char* mac)
{
    struct ifreq ifr;
    struct ifconf ifc;
    char buf[1024];
    int success = 0;

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock == -1) { /* handle error*/ };

    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = buf;
    if (ioctl(sock, SIOCGIFCONF, &ifc) == -1) { /* handle error */ }

    struct ifreq* it = ifc.ifc_req;
    const struct ifreq* const end = it + (ifc.ifc_len / sizeof(struct ifreq));

    for (; it != end; ++it) {
        strcpy(ifr.ifr_name, it->ifr_name);
        if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0) {
            if (!(ifr.ifr_flags & IFF_LOOPBACK)) { // don't count loopback
                if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
                    success = 1;
                    break;
                }
            }
        }
        else { /* handle error */ }
    }
    if (success)
    {
        memcpy(mac, ifr.ifr_hwaddr.sa_data, 6);
        char print_buf[256];
        sprintf(print_buf, "%02x:%02x:%02x:%02x:%02x:%02x\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        std::cout << "Machine MAC " << print_buf << std::endl;
    }
    return success;
}

bool match(unsigned char* addr, unsigned char addr_list[3][6])
{
    for (int a = 0; a < 3; ++a)
    {
        bool matched = true;
        int i = 0;
        for (i = 0; i < 6; i++)
        {
            if (addr[i] != addr_list[a][i])
            {
                matched = false;
                break;
            }
        }
        if (matched)
        {
            return true;
        }
    }
    return false;
}
unsigned char addr_list[3][6] = { {0xac, 0x1f, 0x6b, 0x22, 0xd4, 0xe6 }, {0x02,0x42,0xac,0x11,0x00,0x02 }, {0x02,0x42,0xac,0x11,0x00,0x03 } };
bool GetMACaddress(void)
{
#ifdef USE_INNER_TZTX
    return true;
#endif
    unsigned char mac[6] = { 0 };
    bool success = getMacAddress(mac);
    success = match(mac, addr_list);
    return success;
}

#endif
#endif
