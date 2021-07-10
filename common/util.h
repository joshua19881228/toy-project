#pragma once
#include <string>
#include <map>
#include <strstream>
#include <iostream>

#ifdef __ANDROID__
#include <android/log.h>
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "deeperception", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG , "deeperception", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO  , "deeperception", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN  , "deeperception", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , "deeperception", __VA_ARGS__)
#else
#define LOGV(...) printf("deeperception VERBOSE: ");printf(__VA_ARGS__);printf("\n");
#define LOGD(...) printf("deeperception DEBUG: ");printf(__VA_ARGS__);printf("\n");
#define LOGI(...) printf("deeperception INFO: ");printf(__VA_ARGS__);printf("\n");
#define LOGW(...) printf("deeperception WARNING: ");printf(__VA_ARGS__);printf("\n");
#define LOGE(...) printf("deeperception ERROR: ");printf(__VA_ARGS__);printf("\n");
#endif


using std::string;
using std::map;
using std::strstream;

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT 
#endif

template<typename dtype>
dtype str2num(const string& string_input)
{
    dtype int_temp;
    strstream stream;
    stream << string_input;
    stream >> int_temp;
    return int_temp;
}

template<typename dtype>
string num2str(const dtype& num_input)
{
    strstream stream;
    string string_output;
    stream << num_input;
    stream >> string_output;
    return string_output;
}

int DLLEXPORT loadConfigFile(const string& config_file_path, map<string, string>& configs);
bool DLLEXPORT verifyLicense(const string& license_path);