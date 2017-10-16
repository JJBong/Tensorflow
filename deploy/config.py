packet_flow_files_dir="/Users/jubong/git/Tensorflow/deploy/packet_flow_files_dir/"

process_definition_policy = {
    "web" : ["google chrome","chrome.exe", "iexplore.exe", "firefox", "microsoftedge.exe", "microsoftedgecp.exe", False],
    "dropbox" : ["dropbox", "dropbox.exe", False],
    "kakaotalk" : ["kakaotalk", "kakaotalk.exe", False],
    "slack" : ["slack", "slack.exe", False],
    "mysqlworkbench" : ["mysqlworkbench", "mysqlworkbench.exe", False],
    "teamviewer" : ["teamviewer", "teamviewer.exe", "teamviewer_desktop", "teamviewer_service", "teamviewer_service.exe", "teamviewerd", False],
    "com.apple" : ["com.apple.geod", "com.apple.photomoments", "com.apple.safari.safebrowsing.service", "com.apple.webkit.networking", False],
    "onedrive" : ["onedrive", "onedrive.exe", False],
    "python" : ["python", "python3.6", "python.exe", False],
    "skype" : ["skype", "skype.exe", "skypebrowserhost.exe", "skypehost.exe", False],
    "utorrent" : ["utorrent', 'utorrentie.exe", False],
    "vnc" : ["vncserver", "vncviewer", False],
    "git" : ["git", "git-remote-https.exe", False],
    "leagueoflegends" : ["leagueclient.exe", "leagueclientux.exe", False],
    "wunderlist" : ["wunderlist", "wunderlisthelper", False]
}

#DB information
host = "218.150.181.120"
port = 33060
user = "etri"
password = "linketri"
db = "network"
charset = "utf8"