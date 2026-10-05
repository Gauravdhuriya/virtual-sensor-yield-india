#!/usr/bin/env python3
"""
fetch_india_data.py - download the inputs for the Indian district experiments.

For each of the 311 ICRISAT districts this script
  1. finds the coordinates of the district headquarters (OpenStreetMap Nominatim, Open-Meteo as fallback),
  2. downloads NASA POWER daily weather 1984-2017: rainfall, air temperature, wind speed,
     solar radiation, relative humidity,
  3. looks up topsoil pH (0-30 cm) from ISRIC SoilGrids,
and packs everything into a few small files in the folder india_data/ for upload.

Usage:   pip install requests numpy pandas
         python fetch_india_data.py
It takes about 1-1.5 hours. It is safe to stop and restart: finished districts are cached and skipped.
"""
import io, json, os, sys, time
import numpy as np
import pandas as pd
import requests

OUT, CACHE = "india_data", "india_data/cache"
START, END = "19840101", "20171231"
PARAMS = ["PRECTOTCORR", "T2M", "WS2M", "ALLSKY_SFC_SW_DWN", "RH2M"]
SCALE = [10, 100, 100, 100, 100]          # stored as int16 = value * scale
MISSING = -32768
UA = {"User-Agent": "adt-phd-research/1.0 (district weather download for crop yield study)"}

# rough state bounding boxes (lat_min, lat_max, lon_min, lon_max) used to reject wrong geocoding hits
BBOX = {
    "Chhattisgarh": (17.7, 24.2, 80.2, 84.5), "Madhya Pradesh": (21.0, 26.9, 74.0, 82.9),
    "Andhra Pradesh": (12.6, 19.2, 76.7, 84.8), "Telangana": (15.8, 19.95, 77.2, 81.4),
    "Karnataka": (11.5, 18.5, 74.0, 78.6), "Tamil Nadu": (8.0, 13.6, 76.2, 80.4),
    "Maharashtra": (15.6, 22.1, 72.6, 80.9), "Gujarat": (20.0, 24.75, 68.1, 74.5),
    "Rajasthan": (23.0, 30.2, 69.4, 78.3), "Punjab": (29.5, 32.6, 73.8, 77.0),
    "Haryana": (27.6, 31.0, 74.4, 77.6), "Uttar Pradesh": (23.8, 30.5, 77.0, 84.7),
    "Uttarakhand": (28.7, 31.5, 77.5, 81.1), "Assam": (24.1, 28.0, 89.6, 96.1),
    "Himachal Pradesh": (30.3, 33.3, 75.5, 79.1), "Kerala": (8.1, 12.8, 74.8, 77.5),
    "Orissa": (17.8, 22.6, 81.3, 87.5), "West Bengal": (21.5, 27.3, 85.8, 89.9),
    "Bihar": (24.2, 27.6, 83.3, 88.3), "Jharkhand": (21.9, 25.4, 83.3, 87.9),
}

# (ICRISAT Dist Code, State Name, Dist Name, town used for geocoding, state used for geocoding)
DISTRICTS = [
    (1, "Chhattisgarh", "Durg", "Durg", "Chhattisgarh"),
    (2, "Chhattisgarh", "Bastar", "Jagdalpur", "Chhattisgarh"),
    (3, "Chhattisgarh", "Raipur", "Raipur", "Chhattisgarh"),
    (4, "Chhattisgarh", "Bilaspur", "Bilaspur", "Chhattisgarh"),
    (5, "Chhattisgarh", "Raigarh", "Raigarh", "Chhattisgarh"),
    (6, "Chhattisgarh", "Surguja", "Ambikapur", "Chhattisgarh"),
    (7, "Madhya Pradesh", "Jabalpur", "Jabalpur", "Madhya Pradesh"),
    (8, "Madhya Pradesh", "Balaghat", "Balaghat", "Madhya Pradesh"),
    (9, "Madhya Pradesh", "Chhindwara", "Chhindwara", "Madhya Pradesh"),
    (10, "Madhya Pradesh", "Narsinghpur", "Narsinghpur", "Madhya Pradesh"),
    (11, "Madhya Pradesh", "Seoni / Shivani", "Seoni", "Madhya Pradesh"),
    (12, "Madhya Pradesh", "Mandla", "Mandla", "Madhya Pradesh"),
    (13, "Madhya Pradesh", "Sagar", "Sagar", "Madhya Pradesh"),
    (14, "Madhya Pradesh", "Damoh", "Damoh", "Madhya Pradesh"),
    (15, "Madhya Pradesh", "Tikamgarh", "Tikamgarh", "Madhya Pradesh"),
    (16, "Madhya Pradesh", "Chhatarpur", "Chhatarpur", "Madhya Pradesh"),
    (17, "Madhya Pradesh", "Panna", "Panna", "Madhya Pradesh"),
    (18, "Madhya Pradesh", "Rewa", "Rewa", "Madhya Pradesh"),
    (19, "Madhya Pradesh", "Sidhi", "Sidhi", "Madhya Pradesh"),
    (20, "Madhya Pradesh", "Satna", "Satna", "Madhya Pradesh"),
    (21, "Madhya Pradesh", "Shahdol", "Shahdol", "Madhya Pradesh"),
    (22, "Madhya Pradesh", "Gwalior", "Gwalior", "Madhya Pradesh"),
    (23, "Madhya Pradesh", "Shivpuri", "Shivpuri", "Madhya Pradesh"),
    (24, "Madhya Pradesh", "Guna", "Guna", "Madhya Pradesh"),
    (25, "Madhya Pradesh", "Datia", "Datia", "Madhya Pradesh"),
    (26, "Madhya Pradesh", "Morena", "Morena", "Madhya Pradesh"),
    (27, "Madhya Pradesh", "Bhind", "Bhind", "Madhya Pradesh"),
    (28, "Madhya Pradesh", "Indore", "Indore", "Madhya Pradesh"),
    (29, "Madhya Pradesh", "Ratlam", "Ratlam", "Madhya Pradesh"),
    (30, "Madhya Pradesh", "Ujjain", "Ujjain", "Madhya Pradesh"),
    (31, "Madhya Pradesh", "Mandsaur", "Mandsaur", "Madhya Pradesh"),
    (32, "Madhya Pradesh", "Dewas", "Dewas", "Madhya Pradesh"),
    (33, "Madhya Pradesh", "Dhar", "Dhar", "Madhya Pradesh"),
    (34, "Madhya Pradesh", "Jhabua", "Jhabua", "Madhya Pradesh"),
    (35, "Madhya Pradesh", "Khargone / West Nimar", "Khargone", "Madhya Pradesh"),
    (36, "Madhya Pradesh", "Khandwa / East Nimar", "Khandwa", "Madhya Pradesh"),
    (37, "Madhya Pradesh", "Sehore", "Sehore", "Madhya Pradesh"),
    (38, "Madhya Pradesh", "Raisen", "Raisen", "Madhya Pradesh"),
    (39, "Madhya Pradesh", "Vidisha", "Vidisha", "Madhya Pradesh"),
    (40, "Madhya Pradesh", "Betul", "Betul", "Madhya Pradesh"),
    (41, "Madhya Pradesh", "Rajgarh", "Rajgarh", "Madhya Pradesh"),
    (42, "Madhya Pradesh", "Shajapur", "Shajapur", "Madhya Pradesh"),
    (43, "Madhya Pradesh", "Hoshangabad", "Hoshangabad", "Madhya Pradesh"),
    (44, "Andhra Pradesh", "Srikakulam", "Srikakulam", "Andhra Pradesh"),
    (45, "Andhra Pradesh", "Visakhapatnam", "Visakhapatnam", "Andhra Pradesh"),
    (46, "Andhra Pradesh", "East Godavari", "Kakinada", "Andhra Pradesh"),
    (47, "Andhra Pradesh", "West Godavari", "Eluru", "Andhra Pradesh"),
    (48, "Andhra Pradesh", "Krishna", "Machilipatnam", "Andhra Pradesh"),
    (49, "Andhra Pradesh", "Guntur", "Guntur", "Andhra Pradesh"),
    (50, "Andhra Pradesh", "S.P.S. Nellore", "Nellore", "Andhra Pradesh"),
    (51, "Andhra Pradesh", "Kurnool", "Kurnool", "Andhra Pradesh"),
    (52, "Andhra Pradesh", "Ananthapur", "Anantapur", "Andhra Pradesh"),
    (53, "Andhra Pradesh", "Kadapa YSR", "Kadapa", "Andhra Pradesh"),
    (54, "Andhra Pradesh", "Chittoor", "Chittoor", "Andhra Pradesh"),
    (55, "Telangana", "Hyderabad", "Hyderabad", "Telangana"),
    (56, "Telangana", "Nizamabad", "Nizamabad", "Telangana"),
    (57, "Telangana", "Medak", "Medak", "Telangana"),
    (58, "Telangana", "Mahabubnagar", "Mahabubnagar", "Telangana"),
    (59, "Telangana", "Nalgonda", "Nalgonda", "Telangana"),
    (60, "Telangana", "Warangal", "Warangal", "Telangana"),
    (61, "Telangana", "Khammam", "Khammam", "Telangana"),
    (62, "Telangana", "Karimnagar", "Karimnagar", "Telangana"),
    (63, "Telangana", "Adilabad", "Adilabad", "Telangana"),
    (64, "Karnataka", "Bangalore", "Bengaluru", "Karnataka"),
    (65, "Karnataka", "Kolar", "Kolar", "Karnataka"),
    (66, "Karnataka", "Tumkur", "Tumakuru", "Karnataka"),
    (67, "Karnataka", "Mysore", "Mysuru", "Karnataka"),
    (68, "Karnataka", "Mandya", "Mandya", "Karnataka"),
    (69, "Karnataka", "Hassan", "Hassan", "Karnataka"),
    (70, "Karnataka", "Shimoge", "Shivamogga", "Karnataka"),
    (71, "Karnataka", "Chickmagalur", "Chikkamagaluru", "Karnataka"),
    (72, "Karnataka", "Chitradurga", "Chitradurga", "Karnataka"),
    (73, "Karnataka", "Bellary", "Ballari", "Karnataka"),
    (74, "Karnataka", "Dharwad", "Dharwad", "Karnataka"),
    (75, "Karnataka", "Belgaum", "Belagavi", "Karnataka"),
    (76, "Karnataka", "Bijapur / Vijayapura", "Vijayapura", "Karnataka"),
    (77, "Karnataka", "Bidar", "Bidar", "Karnataka"),
    (78, "Karnataka", "Raichur", "Raichur", "Karnataka"),
    (79, "Karnataka", "Gulbarga / Kalaburagi", "Kalaburagi", "Karnataka"),
    (80, "Karnataka", "Dakshina Kannada", "Mangaluru", "Karnataka"),
    (81, "Karnataka", "Uttara Kannada", "Karwar", "Karnataka"),
    (82, "Karnataka", "Kodagu / Coorg", "Madikeri", "Karnataka"),
    (83, "Tamil Nadu", "Chengalpattu MGR / Kanchipuram", "Kanchipuram", "Tamil Nadu"),
    (84, "Tamil Nadu", "South Arcot / Cuddalore", "Cuddalore", "Tamil Nadu"),
    (85, "Tamil Nadu", "North Arcot / Vellore", "Vellore", "Tamil Nadu"),
    (86, "Tamil Nadu", "Salem", "Salem", "Tamil Nadu"),
    (87, "Tamil Nadu", "Coimbatore", "Coimbatore", "Tamil Nadu"),
    (88, "Tamil Nadu", "Tiruchirapalli / Trichy", "Tiruchirappalli", "Tamil Nadu"),
    (89, "Tamil Nadu", "Thanjavur", "Thanjavur", "Tamil Nadu"),
    (90, "Tamil Nadu", "Madurai", "Madurai", "Tamil Nadu"),
    (91, "Tamil Nadu", "Ramananthapuram", "Ramanathapuram", "Tamil Nadu"),
    (92, "Tamil Nadu", "Thirunelveli", "Tirunelveli", "Tamil Nadu"),
    (93, "Tamil Nadu", "The Nilgiris", "Udhagamandalam", "Tamil Nadu"),
    (94, "Tamil Nadu", "Kanyakumari", "Nagercoil", "Tamil Nadu"),
    (95, "Maharashtra", "Bombay", "Mumbai", "Maharashtra"),
    (96, "Maharashtra", "Thane", "Thane", "Maharashtra"),
    (97, "Maharashtra", "Raigad", "Alibag", "Maharashtra"),
    (98, "Maharashtra", "Ratnagiri", "Ratnagiri", "Maharashtra"),
    (99, "Maharashtra", "Nasik", "Nashik", "Maharashtra"),
    (100, "Maharashtra", "Dhule", "Dhule", "Maharashtra"),
    (101, "Maharashtra", "Jalgaon", "Jalgaon", "Maharashtra"),
    (102, "Maharashtra", "Ahmednagar", "Ahmednagar", "Maharashtra"),
    (103, "Maharashtra", "Pune", "Pune", "Maharashtra"),
    (104, "Maharashtra", "Satara", "Satara", "Maharashtra"),
    (105, "Maharashtra", "Sangli", "Sangli", "Maharashtra"),
    (106, "Maharashtra", "Solapur", "Solapur", "Maharashtra"),
    (107, "Maharashtra", "Kolhapur", "Kolhapur", "Maharashtra"),
    (108, "Maharashtra", "Aurangabad", "Aurangabad", "Maharashtra"),
    (109, "Maharashtra", "Parbhani", "Parbhani", "Maharashtra"),
    (110, "Maharashtra", "Beed", "Beed", "Maharashtra"),
    (111, "Maharashtra", "Nanded", "Nanded", "Maharashtra"),
    (112, "Maharashtra", "Osmanabad", "Osmanabad", "Maharashtra"),
    (113, "Maharashtra", "Buldhana", "Buldhana", "Maharashtra"),
    (114, "Maharashtra", "Akola", "Akola", "Maharashtra"),
    (115, "Maharashtra", "Amarawati", "Amravati", "Maharashtra"),
    (116, "Maharashtra", "Yeotmal", "Yavatmal", "Maharashtra"),
    (117, "Maharashtra", "Wardha", "Wardha", "Maharashtra"),
    (118, "Maharashtra", "Nagpur", "Nagpur", "Maharashtra"),
    (119, "Maharashtra", "Bhandara", "Bhandara", "Maharashtra"),
    (120, "Maharashtra", "Chandrapur", "Chandrapur", "Maharashtra"),
    (121, "Gujarat", "Ahmedabad", "Ahmedabad", "Gujarat"),
    (122, "Gujarat", "Amreli", "Amreli", "Gujarat"),
    (123, "Gujarat", "Banaskantha", "Palanpur", "Gujarat"),
    (124, "Gujarat", "Bharuch", "Bharuch", "Gujarat"),
    (125, "Gujarat", "Vadodara / Baroda", "Vadodara", "Gujarat"),
    (126, "Gujarat", "Bhavnagar", "Bhavnagar", "Gujarat"),
    (127, "Gujarat", "Valsad", "Valsad", "Gujarat"),
    (128, "Gujarat", "Dangs", "Ahwa", "Gujarat"),
    (129, "Gujarat", "Jamnagar", "Jamnagar", "Gujarat"),
    (130, "Gujarat", "Junagadh", "Junagadh", "Gujarat"),
    (131, "Gujarat", "Kheda", "Nadiad", "Gujarat"),
    (132, "Gujarat", "Kutch", "Bhuj", "Gujarat"),
    (133, "Gujarat", "Mehsana", "Mehsana", "Gujarat"),
    (134, "Gujarat", "Panchmahal", "Godhra", "Gujarat"),
    (135, "Gujarat", "Rajkot", "Rajkot", "Gujarat"),
    (136, "Gujarat", "Sabarkantha", "Himatnagar", "Gujarat"),
    (137, "Gujarat", "Surat", "Surat", "Gujarat"),
    (138, "Gujarat", "Surendranagar", "Surendranagar", "Gujarat"),
    (139, "Rajasthan", "Ajmer", "Ajmer", "Rajasthan"),
    (140, "Rajasthan", "Alwar", "Alwar", "Rajasthan"),
    (141, "Rajasthan", "Banswara", "Banswara", "Rajasthan"),
    (142, "Rajasthan", "Barmer", "Barmer", "Rajasthan"),
    (143, "Rajasthan", "Bharatpur", "Bharatpur", "Rajasthan"),
    (144, "Rajasthan", "Bhilwara", "Bhilwara", "Rajasthan"),
    (145, "Rajasthan", "Bikaner", "Bikaner", "Rajasthan"),
    (146, "Rajasthan", "Bundi", "Bundi", "Rajasthan"),
    (147, "Rajasthan", "Chittorgarh", "Chittorgarh", "Rajasthan"),
    (148, "Rajasthan", "Churu", "Churu", "Rajasthan"),
    (149, "Rajasthan", "Dungarpur", "Dungarpur", "Rajasthan"),
    (150, "Rajasthan", "Ganganagar", "Sri Ganganagar", "Rajasthan"),
    (151, "Rajasthan", "Jaipur", "Jaipur", "Rajasthan"),
    (152, "Rajasthan", "Jaisalmer", "Jaisalmer", "Rajasthan"),
    (153, "Rajasthan", "Jalore", "Jalore", "Rajasthan"),
    (154, "Rajasthan", "Jhalawar", "Jhalawar", "Rajasthan"),
    (155, "Rajasthan", "Jhunjhunu", "Jhunjhunu", "Rajasthan"),
    (156, "Rajasthan", "Jodhpur", "Jodhpur", "Rajasthan"),
    (157, "Rajasthan", "Kota", "Kota", "Rajasthan"),
    (158, "Rajasthan", "Nagaur", "Nagaur", "Rajasthan"),
    (159, "Rajasthan", "Pali", "Pali", "Rajasthan"),
    (160, "Rajasthan", "Swami Madhopur", "Sawai Madhopur", "Rajasthan"),
    (161, "Rajasthan", "Sikar", "Sikar", "Rajasthan"),
    (162, "Rajasthan", "Sirohi", "Sirohi", "Rajasthan"),
    (163, "Rajasthan", "Tonk", "Tonk", "Rajasthan"),
    (164, "Rajasthan", "Udaipur", "Udaipur", "Rajasthan"),
    (165, "Punjab", "Gurdaspur", "Gurdaspur", "Punjab"),
    (166, "Punjab", "Amritsar", "Amritsar", "Punjab"),
    (167, "Punjab", "Kapurthala", "Kapurthala", "Punjab"),
    (168, "Punjab", "Jalandhar", "Jalandhar", "Punjab"),
    (169, "Punjab", "Hoshiarpur", "Hoshiarpur", "Punjab"),
    (170, "Punjab", "Roopnagar / Ropar", "Rupnagar", "Punjab"),
    (171, "Punjab", "Ludhiana", "Ludhiana", "Punjab"),
    (172, "Punjab", "Ferozpur", "Firozpur", "Punjab"),
    (174, "Punjab", "Bhatinda", "Bathinda", "Punjab"),
    (175, "Punjab", "Sangrur", "Sangrur", "Punjab"),
    (176, "Punjab", "Patiala", "Patiala", "Punjab"),
    (177, "Haryana", "Hissar", "Hisar", "Haryana"),
    (180, "Haryana", "Gurgaon", "Gurugram", "Haryana"),
    (181, "Haryana", "Jind", "Jind", "Haryana"),
    (182, "Haryana", "Mahendragarh / Narnaul", "Narnaul", "Haryana"),
    (183, "Haryana", "Ambala", "Ambala", "Haryana"),
    (184, "Haryana", "Karnal", "Karnal", "Haryana"),
    (186, "Haryana", "Rohtak", "Rohtak", "Haryana"),
    (189, "Uttar Pradesh", "Saharanpur", "Saharanpur", "Uttar Pradesh"),
    (190, "Uttar Pradesh", "Muzaffarnagar", "Muzaffarnagar", "Uttar Pradesh"),
    (191, "Uttar Pradesh", "Meerut", "Meerut", "Uttar Pradesh"),
    (192, "Uttar Pradesh", "Buland Shahar", "Bulandshahr", "Uttar Pradesh"),
    (194, "Uttar Pradesh", "Aligarh", "Aligarh", "Uttar Pradesh"),
    (195, "Uttar Pradesh", "Mathura", "Mathura", "Uttar Pradesh"),
    (196, "Uttar Pradesh", "Agra", "Agra", "Uttar Pradesh"),
    (197, "Uttar Pradesh", "Mainpuri", "Mainpuri", "Uttar Pradesh"),
    (198, "Uttar Pradesh", "Etah", "Etah", "Uttar Pradesh"),
    (199, "Uttar Pradesh", "Bareilly", "Bareilly", "Uttar Pradesh"),
    (200, "Uttar Pradesh", "Budaun", "Budaun", "Uttar Pradesh"),
    (201, "Uttar Pradesh", "Moradabad", "Moradabad", "Uttar Pradesh"),
    (202, "Uttar Pradesh", "Shahjahanpur", "Shahjahanpur", "Uttar Pradesh"),
    (203, "Uttar Pradesh", "Pilibhit", "Pilibhit", "Uttar Pradesh"),
    (204, "Uttar Pradesh", "Rampur", "Rampur", "Uttar Pradesh"),
    (205, "Uttar Pradesh", "Bijnor", "Bijnor", "Uttar Pradesh"),
    (206, "Uttar Pradesh", "Farrukhabad", "Farrukhabad", "Uttar Pradesh"),
    (207, "Uttar Pradesh", "Etawah", "Etawah", "Uttar Pradesh"),
    (208, "Uttar Pradesh", "Kanpur", "Kanpur", "Uttar Pradesh"),
    (209, "Uttar Pradesh", "Fatehpur", "Fatehpur", "Uttar Pradesh"),
    (210, "Uttar Pradesh", "Allahabad", "Prayagraj", "Uttar Pradesh"),
    (211, "Uttar Pradesh", "Jhansi", "Jhansi", "Uttar Pradesh"),
    (213, "Uttar Pradesh", "Jalaun", "Jalaun", "Uttar Pradesh"),
    (214, "Uttar Pradesh", "Hamirpur", "Hamirpur", "Uttar Pradesh"),
    (215, "Uttar Pradesh", "Banda", "Banda", "Uttar Pradesh"),
    (216, "Uttar Pradesh", "Varanasi", "Varanasi", "Uttar Pradesh"),
    (217, "Uttar Pradesh", "Mirzpur", "Mirzapur", "Uttar Pradesh"),
    (218, "Uttar Pradesh", "Jaunpur", "Jaunpur", "Uttar Pradesh"),
    (219, "Uttar Pradesh", "Ghazipur", "Ghazipur", "Uttar Pradesh"),
    (220, "Uttar Pradesh", "Ballia", "Ballia", "Uttar Pradesh"),
    (221, "Uttar Pradesh", "Gorakhpur", "Gorakhpur", "Uttar Pradesh"),
    (222, "Uttar Pradesh", "Deoria", "Deoria", "Uttar Pradesh"),
    (223, "Uttar Pradesh", "Basti", "Basti", "Uttar Pradesh"),
    (224, "Uttar Pradesh", "Azamgarh", "Azamgarh", "Uttar Pradesh"),
    (225, "Uttar Pradesh", "Lucknow", "Lucknow", "Uttar Pradesh"),
    (226, "Uttar Pradesh", "Unnao", "Unnao", "Uttar Pradesh"),
    (227, "Uttar Pradesh", "Rae-Bareily", "Raebareli", "Uttar Pradesh"),
    (228, "Uttar Pradesh", "Sitapur", "Sitapur", "Uttar Pradesh"),
    (229, "Uttar Pradesh", "Hardoi", "Hardoi", "Uttar Pradesh"),
    (230, "Uttar Pradesh", "Kheri", "Lakhimpur", "Uttar Pradesh"),
    (231, "Uttar Pradesh", "Faizabad", "Faizabad", "Uttar Pradesh"),
    (232, "Uttar Pradesh", "Gonda", "Gonda", "Uttar Pradesh"),
    (233, "Uttar Pradesh", "Bahraich", "Bahraich", "Uttar Pradesh"),
    (234, "Uttar Pradesh", "Sultanpur", "Sultanpur", "Uttar Pradesh"),
    (235, "Uttar Pradesh", "Pratapgarh", "Pratapgarh", "Uttar Pradesh"),
    (236, "Uttar Pradesh", "Barabanki", "Barabanki", "Uttar Pradesh"),
    (237, "Uttarakhand", "Nainital", "Nainital", "Uttarakhand"),
    (238, "Uttarakhand", "Almorah", "Almora", "Uttarakhand"),
    (239, "Uttarakhand", "Pithorgarh", "Pithoragarh", "Uttarakhand"),
    (240, "Uttarakhand", "Chamoli", "Gopeshwar", "Uttarakhand"),
    (241, "Uttarakhand", "Uttar Kashi", "Uttarkashi", "Uttarakhand"),
    (242, "Uttarakhand", "Tehri Garhwal", "New Tehri", "Uttarakhand"),
    (243, "Uttarakhand", "Garhwal", "Pauri", "Uttarakhand"),
    (244, "Uttarakhand", "Dehradun", "Dehradun", "Uttarakhand"),
    (601, "Assam", "Cachar", "Silchar", "Assam"),
    (602, "Assam", "Darrang", "Mangaldoi", "Assam"),
    (603, "Assam", "Dibrugarh", "Dibrugarh", "Assam"),
    (604, "Assam", "Goalpara", "Goalpara", "Assam"),
    (605, "Assam", "Kamrup", "Guwahati", "Assam"),
    (606, "Assam", "Karbi Anglong", "Diphu", "Assam"),
    (607, "Assam", "Lakhimpur", "North Lakhimpur", "Assam"),
    (608, "Assam", "North Cachar Hil / Dima hasao", "Haflong", "Assam"),
    (609, "Assam", "Nagaon", "Nagaon", "Assam"),
    (610, "Assam", "Sibsagar", "Sivasagar", "Assam"),
    (626, "Himachal Pradesh", "Bilashpur", "Bilaspur", "Himachal Pradesh"),
    (627, "Himachal Pradesh", "Chamba", "Chamba", "Himachal Pradesh"),
    (629, "Himachal Pradesh", "Kangra", "Dharamshala", "Himachal Pradesh"),
    (630, "Himachal Pradesh", "Kinnaur", "Reckong Peo", "Himachal Pradesh"),
    (631, "Himachal Pradesh", "Kullu", "Kullu", "Himachal Pradesh"),
    (632, "Himachal Pradesh", "Lahul & Spiti", "Keylong", "Himachal Pradesh"),
    (633, "Himachal Pradesh", "Solan", "Solan", "Himachal Pradesh"),
    (634, "Himachal Pradesh", "Mandi", "Mandi", "Himachal Pradesh"),
    (635, "Himachal Pradesh", "Shimla", "Shimla", "Himachal Pradesh"),
    (636, "Himachal Pradesh", "Sirmaur", "Nahan", "Himachal Pradesh"),
    (651, "Kerala", "Alappuzha", "Alappuzha", "Kerala"),
    (652, "Kerala", "Kannur", "Kannur", "Kerala"),
    (653, "Kerala", "Eranakulam", "Kochi", "Kerala"),
    (655, "Kerala", "Kottayam", "Kottayam", "Kerala"),
    (656, "Kerala", "Kozhikode", "Kozhikode", "Kerala"),
    (657, "Kerala", "Malappuram", "Malappuram", "Kerala"),
    (658, "Kerala", "Palakkad", "Palakkad", "Kerala"),
    (659, "Kerala", "Kollam", "Kollam", "Kerala"),
    (660, "Kerala", "Thrissur", "Thrissur", "Kerala"),
    (661, "Kerala", "Thiruvananthapuram", "Thiruvananthapuram", "Kerala"),
    (701, "Orissa", "Balasore", "Balasore", "Odisha"),
    (702, "Orissa", "Bolangir", "Balangir", "Odisha"),
    (703, "Orissa", "Cuttack", "Cuttack", "Odisha"),
    (704, "Orissa", "Dhenkanal", "Dhenkanal", "Odisha"),
    (705, "Orissa", "Ganjam", "Brahmapur", "Odisha"),
    (706, "Orissa", "Kalahandi", "Bhawanipatna", "Odisha"),
    (707, "Orissa", "Keonjhar", "Keonjhar", "Odisha"),
    (708, "Orissa", "Koraput", "Koraput", "Odisha"),
    (709, "Orissa", "Mayurbhanja", "Baripada", "Odisha"),
    (710, "Orissa", "Phulbani ( Kandhamal )", "Phulbani", "Odisha"),
    (711, "Orissa", "Puri", "Puri", "Odisha"),
    (712, "Orissa", "Sambalpur", "Sambalpur", "Odisha"),
    (713, "Orissa", "Sundargarh", "Sundargarh", "Odisha"),
    (800, "West Bengal", "24 Parganas", "Barasat", "West Bengal"),
    (803, "West Bengal", "Nadia", "Krishnanagar", "West Bengal"),
    (804, "West Bengal", "Murshidabad", "Baharampur", "West Bengal"),
    (805, "West Bengal", "Burdwan", "Bardhaman", "West Bengal"),
    (806, "West Bengal", "Birbhum", "Suri", "West Bengal"),
    (807, "West Bengal", "Bankura", "Bankura", "West Bengal"),
    (810, "West Bengal", "Hooghly", "Chinsurah", "West Bengal"),
    (811, "West Bengal", "Howrah", "Howrah", "West Bengal"),
    (812, "West Bengal", "Jalpaiguri", "Jalpaiguri", "West Bengal"),
    (813, "West Bengal", "Darjeeling", "Darjeeling", "West Bengal"),
    (814, "West Bengal", "Malda", "English Bazar", "West Bengal"),
    (816, "West Bengal", "Cooch Behar", "Cooch Behar", "West Bengal"),
    (817, "West Bengal", "Purulia", "Purulia", "West Bengal"),
    (818, "West Bengal", "Midnapur", "Medinipur", "West Bengal"),
    (822, "West Bengal", "West Dinajpur", "Balurghat", "West Bengal"),
    (901, "Bihar", "Champaran", "Motihari", "Bihar"),
    (902, "Bihar", "Muzaffarpur", "Muzaffarpur", "Bihar"),
    (903, "Bihar", "Darbhanga", "Darbhanga", "Bihar"),
    (904, "Bihar", "Saharsa", "Saharsa", "Bihar"),
    (905, "Bihar", "Purnea", "Purnia", "Bihar"),
    (906, "Bihar", "Saran", "Chhapra", "Bihar"),
    (907, "Bihar", "Patna", "Patna", "Bihar"),
    (908, "Bihar", "Mungair", "Munger", "Bihar"),
    (909, "Bihar", "Bhagalpur", "Bhagalpur", "Bihar"),
    (910, "Jharkhand", "Santhal Paragana / Dumka", "Dumka", "Jharkhand"),
    (911, "Bihar", "Shahabad (now part of Bhojpur district)", "Arrah", "Bihar"),
    (912, "Bihar", "Gaya", "Gaya", "Bihar"),
    (913, "Jharkhand", "Hazaribagh", "Hazaribagh", "Jharkhand"),
    (914, "Jharkhand", "Dhanbad", "Dhanbad", "Jharkhand"),
    (915, "Jharkhand", "Palamau", "Medininagar", "Jharkhand"),
    (916, "Jharkhand", "Ranchi", "Ranchi", "Jharkhand"),
    (917, "Jharkhand", "Singhbhum", "Chaibasa", "Jharkhand"),
]


def get(url, params=None, tries=5, timeout=120, headers=UA):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}"
        except Exception as e:                       # network hiccup: wait and retry
            err = repr(e)[:80]
        time.sleep(5 * (k + 1))
    print(f"      failed: {err}")
    return None


def inside(state, lat, lon, m=0.2):
    a, b, c, d = BBOX[state]
    return a - m <= lat <= b + m and c - m <= lon <= d + m


def geocode(state, town, qstate):
    r = get("https://nominatim.openstreetmap.org/search",
            dict(q=f"{town}, {qstate}, India", format="jsonv2", limit=5, countrycodes="in"), tries=3, timeout=30)
    time.sleep(1.1)                                  # Nominatim usage policy: max 1 request per second
    if r is not None:
        for h in r.json():
            lat, lon = float(h["lat"]), float(h["lon"])
            if inside(state, lat, lon):
                return lat, lon, "nominatim"
    r = get("https://geocoding-api.open-meteo.com/v1/search",
            dict(name=town, count=10, language="en", format="json", countryCode="IN"), tries=3, timeout=30)
    if r is not None:
        for h in r.json().get("results", []) or []:
            if inside(state, h["latitude"], h["longitude"]):
                return h["latitude"], h["longitude"], "open-meteo"
    return np.nan, np.nan, "not found"


def parse_power(js):
    """NASA POWER daily JSON -> (dates, int16 array [days, variables])."""
    par = js["properties"]["parameter"]
    dates = sorted(par[PARAMS[0]].keys())
    arr = np.full((len(dates), len(PARAMS)), MISSING, np.int16)
    for j, (p, s) in enumerate(zip(PARAMS, SCALE)):
        v = np.array([par[p].get(d, -999) for d in dates], float)
        ok = v > -900                                # POWER fill value is -999
        arr[ok, j] = np.clip(np.round(v[ok] * s), -32767, 32767).astype(np.int16)
    return np.array(dates), arr


def power(lat, lon):
    r = get("https://power.larc.nasa.gov/api/temporal/daily/point",
            dict(parameters=",".join(PARAMS), community="AG", latitude=round(lat, 4), longitude=round(lon, 4),
                 start=START, end=END, format="JSON"))
    return None if r is None else parse_power(r.json())


def parse_soilgrids(js):
    vals = []
    for layer in js["properties"]["layers"]:
        f = layer["unit_measure"]["d_factor"]
        vals += [d["values"]["mean"] / f for d in layer["depths"] if d["values"]["mean"] is not None]
    return float(np.mean(vals)) if vals else np.nan


_last_soil = [0.0]
def soil_ph(lat, lon):
    for dlat, dlon in ((0, 0), (0.05, 0.05), (-0.05, -0.05), (0.1, -0.1)):   # town centres can be masked
        wait = 13 - (time.time() - _last_soil[0])                              # SoilGrids: about 5 calls/minute
        if wait > 0:
            time.sleep(wait)
        _last_soil[0] = time.time()
        r = get("https://rest.isric.org/soilgrids/v2.0/properties/query",
                [("lon", round(lon + dlon, 4)), ("lat", round(lat + dlat, 4)), ("property", "phh2o"),
                 ("depth", "0-5cm"), ("depth", "5-15cm"), ("depth", "15-30cm"), ("value", "mean")], tries=2, timeout=60)
        if r is None:
            return np.nan
        ph = parse_soilgrids(r.json())
        if not np.isnan(ph):
            return ph
    return np.nan


def main():
    os.makedirs(CACHE, exist_ok=True)
    meta_path = f"{CACHE}/meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    t0 = time.time()
    for i, (code, state, name, town, qstate) in enumerate(DISTRICTS):
        m = meta.setdefault(str(code), {})
        if "lat" not in m or m.get("geo_source") == "not found":
            m["lat"], m["lon"], m["geo_source"] = geocode(state, town, qstate)
        npy = f"{CACHE}/{code}.npy"
        status = "cached"
        if not np.isnan(m["lat"]) and not os.path.exists(npy):
            res = power(m["lat"], m["lon"])
            if res is not None:
                dates, arr = res
                np.save(npy, arr)
                np.save(f"{CACHE}/dates.npy", dates)
                status = "downloaded"
            else:
                status = "WEATHER FAILED"
        if not np.isnan(m["lat"]) and m.get("ph") is None:
            ph = soil_ph(m["lat"], m["lon"])
            m["ph"] = None if np.isnan(ph) else round(ph, 2)
        json.dump(meta, open(meta_path, "w"))
        print(f"[{i+1:3d}/{len(DISTRICTS)}] {name[:28]:28s} {state[:16]:16s} "
              f"lat {m['lat']:.2f} lon {m['lon']:.2f} ({m['geo_source']})  weather: {status}  pH: {m.get('ph')}  "
              f"[{(time.time()-t0)/60:.0f} min]", flush=True)
    pack(meta)


def pack(meta):
    rows = []
    for code, state, name, town, qstate in DISTRICTS:
        m = meta.get(str(code), {})
        rows.append(dict(dist_code=code, state=state, district=name, town=town, lat=m.get("lat"), lon=m.get("lon"),
                         geo_source=m.get("geo_source"), soil_ph=m.get("ph"),
                         has_weather=os.path.exists(f"{CACHE}/{code}.npy")))
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/india_districts.csv", index=False)
    have = tab.loc[tab["has_weather"], "dist_code"].tolist()
    if not have:
        print("No weather downloaded - check your internet connection and run again."); return
    dates = np.load(f"{CACHE}/dates.npy")
    for p, k in enumerate(range(0, len(have), 110)):
        codes = have[k:k + 110]
        data = np.stack([np.load(f"{CACHE}/{c}.npy") for c in codes])
        np.savez_compressed(f"{OUT}/india_weather_part{p+1}.npz", data=data, codes=np.array(codes), dates=dates,
                            params=np.array(PARAMS), scale=np.array(SCALE), missing=MISSING)
    files = sorted(f for f in os.listdir(OUT) if f.endswith((".npz", ".csv")))
    print("\nDone.", len(have), "of", len(tab), "districts have weather;",
          int(tab["soil_ph"].notna().sum()), "have soil pH;", int((tab["geo_source"] == "not found").sum()), "not located.")
    print("Upload these files from the folder", OUT + ":")
    for f in files:
        print(f"   {f}  ({os.path.getsize(os.path.join(OUT, f))/1e6:.1f} MB)")
    if len(have) < len(tab):
        print("Some districts are missing. Running the script again will retry only those.")


if __name__ == "__main__":
    main()
