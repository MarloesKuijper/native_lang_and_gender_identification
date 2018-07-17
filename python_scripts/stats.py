import os
import gender_guesser.detector as gd
import pandas as pd 
import twitter
import time
import re
# def clean_twitter_files(path):
#     folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
#     for folder in folders:
#         print(folder)
#         contents = os.listdir(os.path.join(path,folder))
#         for file in contents:
#             if file.endswith("_messages.txt"):
#                 with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
#                     data = infile.read().split("\n\n")

def clean_medium_files(path):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for folder in folders:
        print(folder)
        contents = os.listdir(os.path.join(path,folder))
        for file in contents:
            if file.startswith("messages_"):
                with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                    data = infile.read().split("\n\n")
                    for item in data:
                        cleanr = re.compile('<.*?>')
                        cleantext = re.sub(cleanr, '', item)
                        if cleantext.split(" ") > 10 and detect(cleantext) == "en":
                            ### keep the data
                            pass

def filter_metadata(path, metadata_filename, new_meta_name):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for folder in folders:
        print(folder)
        contents = os.listdir(os.path.join(path,folder))
        new_meta = []
        for file in contents:
            if file == metadata_filename:
                with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                    data = infile.read().split("\n\n")
                    for item in data:
                        if item:
                            user = item.split("\n")
                            username = user[0].split(":")[-1]
                            if path == "./medium/":
                                file_name = "messages_"+username.strip() + ".txt"
                            elif path == "./twitter/":
                                file_name = username.strip() + "_messages.txt"
                            if file_name in contents and item not in new_meta:
                                new_meta.append(item)
        print("Length: ", len(new_meta))
        with open(path+"/"+folder+"/"+new_meta_name, "w", encoding="utf-8") as outfile:
            for line in new_meta:
                outfile.write(line)
                outfile.write("\n\n")


def create_excel_medium(path, metadata_filename):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    guesser = gd.Detector(case_sensitive=False)
    mapping_dict = {"germany": {"native_lang": "de", "country":"germany"}, "iran":{"native_lang":"farsi", "country":"iran"}, "italy": {"native_lang": "it", "country":"italy"},
                    "new-delhi":{"native_lang": "hindi", "country": "india"}, "poland": {"native_lang": "po", "country":"poland"}, "portugal":{"native_lang": "pt", "country": "portugal"},
                    "russia":{"native_lang": "ru", "country":"russia"}, "spain":{"native_lang": "es", "country": "spain"}, "the-netherlands":{"native_lang":"nl", "country":"the_netherlands"} }

    ## data:
    usernames = []
    names = []
    descriptions = []
    native_langs = []
    genders = []
    tokens = []
    types = []
    for folder in folders:
        print(folder)
        contents = os.listdir(os.path.join(path,folder))
        native_lang = mapping_dict[folder]["native_lang"]
        country = mapping_dict[folder]["country"]
        print(native_lang, country)
        #contents = os.listdir(type_path)
        print(contents)
        for file in contents:
            ## or "metadata_users_new.txt" for medium!!!!
            if file == metadata_filename:
                with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                    data = infile.read().split("\n\n")
                    for item in data:
                        if item:
                            user = item.split("\n")
                            name = user[1].split(":")[-1].strip()
                            description = user[2]
                            username = user[0].split(":")[-1].strip()
                            file_name = "messages_"+username.strip() + ".txt"
                            if file_name in contents and username not in usernames:
                                with open(path+folder+"/"+file_name, "r", encoding="utf-8") as file_contents:
                                    ## TO DO remove tags
                                    data = file_contents.read().split(" ")
                                    tokens.append(len(data))
                                    types.append(len(set(data)))
                                    usernames.append(username)
                                    if description:
                                        descriptions.append(description)
                                    else:
                                        descriptions.append("")
                                    names.append(name)
                                    if len(name.split()) == 2:
                                        first_name = name.split()[0]
                                    else:
                                        first_name = name
                                    try:
                                        gender = guesser.get_gender(first_name, country)
                                    except gd.NoCountryError:
                                        gender = guesser.get_gender(first_name)
                                    genders.append(gender)
                                    native_langs.append(native_lang)


    df = pd.DataFrame({"Username": usernames, "Name": names, "Native language": native_langs, "Gender": genders, "Tokens": tokens, "Types": types, "Description": descriptions})
    df = df[["Username", "Name", "Native language", "Gender", "Tokens", "Types", "Description"]]
    writer = pd.ExcelWriter("data_medium_excel_new_tokens.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Sheet 1", index=False)
    ## TO DO:
    ### check if the texts are actually in English
    ### remove html markup
    ### do manual gender search for the unknown ones
    ### remove duplicates (check if username in username list)

def create_excel_twitter(path, metadata_filename):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    guesser = gd.Detector(case_sensitive=False)
    mapping_dict = {"germany": {"native_lang": "de", "country":"germany"}, "iran":{"native_lang":"farsi", "country":"iran"}, "italy": {"native_lang": "it", "country":"italy"},
                    "new-delhi":{"native_lang": "hindi", "country": "india"}, "poland": {"native_lang": "po", "country":"poland"}, "portugal":{"native_lang": "pt", "country": "portugal"},
                    "russia":{"native_lang": "ru", "country":"russia"}, "spain":{"native_lang": "es", "country": "spain"}, "the-netherlands":{"native_lang":"nl", "country":"the_netherlands"} }

    ## data:
    usernames = []
    names = []
    descriptions = []
    native_langs = []
    genders = []
    tokens = []
    types = []
    for folder in folders:
        print(folder)
        contents = os.listdir(os.path.join(path,folder))
        native_lang = mapping_dict[folder]["native_lang"]
        country = mapping_dict[folder]["country"]
        print(native_lang, country)
        #contents = os.listdir(type_path)
        print(contents)
        for file in contents:
            if file == metadata_filename:
                with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                    data = infile.read().split("\n\n")
                    for item in data:
                        if item:
                            user = item.split("\n")
                            username = user[0].split(":")[-1].strip()
                            
                            if len(user) > 1:
                                description = user[1].split(":")[-1].strip()
                            else:
                                description = ""
                            
                            file_name = username.strip() + "_messages.txt"
                            if file_name in contents and username not in usernames:
                                try:
                                    user_description = api.GetUser(screen_name=username, return_json=True)
                                    #print(user_description)
                                    name = user_description["name"]
                                    print(name)
                                except:
                                    user_description = ""
                                    name = ""
                                with open(path+folder+"/"+file_name, "r", encoding="utf-8") as file_contents:
                                    ## TO DO remove tags
                                    data = file_contents.read().split(" ")
                                    tokens.append(len(data))
                                    types.append(len(set(data)))
                                    usernames.append(username)
                                    descriptions.append(description)
                                    
                                    names.append(name)
                                    # if len(name.split()) == 2:
                                    #     first_name = name.split()[0]
                                    # else:
                                    #     first_name = name
                                    if len(name.split(" ")) > 1:
                                        first_name = name.split(" ")[0]
                                    else:
                                        first_name = name
                                    try:
                                        gender = guesser.get_gender(first_name, country)
                                    except gd.NoCountryError:
                                        gender = guesser.get_gender(first_name)
                                    genders.append(gender)
                                    native_langs.append(native_lang)


    df = pd.DataFrame({"Username": usernames, "Name": names, "Native language": native_langs, "Gender": genders, "Tokens": tokens, "Types": types, "Description": descriptions})
    df = df[["Username", "Name", "Native language", "Gender", "Tokens", "Types", "Description"]]
    writer = pd.ExcelWriter("data_twitter_excel_new_tokens.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Sheet 1", index=False)
    ## TO DO:
    ### check if the texts are actually in English
    ### remove html markup
    ### do manual gender search for the unknown ones
    ### remove duplicates (check if username in username list)
                            

def get_metadata_file_medium(type_path, metadata_filename):
    #folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    # for folder in folders:
    #   print(folder)
    #contents = os.listdir(os.path.join(path,folder))
    contents = os.listdir(type_path)
    print(contents)
    for file in contents:
        ## or "metadata_users_new.txt" for medium!!!!
        if file == metadata_filename:
            with open(os.path.join(type_path, file), "r", encoding="utf-8") as infile:
                data = infile.read().split("\n\n")
                for item in data:
                    if item:
                        user = item.split("\n")
                        description = user[2]
                        username = user[0].split(":")[-1]
                        file_name = "messages_"+username.strip() + ".txt"
                        if file_name in contents:
                            print(description)
                            approved = input("Is this description good enough? (y/n)")
                            if approved == "y":
                                with open(type_path+"approved_users.txt", "a", encoding="utf-8") as outfile:
                                    outfile.write(username)
                                    outfile.write("\n")
                            else:
                                with open(type_path+"rejected_users.txt", "a", encoding="utf-8") as outfile:
                                    outfile.write(username)
                                    outfile.write("\n")

def get_metadata_file_twitter(type_path, metadata_filename):
    #folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    # for folder in folders:
    #   print(folder)
    #contents = os.listdir(os.path.join(path,folder))
    contents = os.listdir(type_path)
    print(contents)
    for file in contents:
        ## or "metadata_users_new.txt" for medium!!!!
        if file == metadata_filename:
            with open(os.path.join(type_path, file), "r", encoding="utf-8") as infile:
                data = infile.read().split("\n\n")
                for item in data:
                    if item:
                        user = item.split("\n")
                        if len(user) > 1:
                            description = user[1]
                            username = user[0].split(":")[-1]
                            file_name = username.strip() + "_messages.txt"
                            if file_name in contents:
                                print(description)
                                approved = input("Is this description good enough? (y/n)")
                                if approved == "y":
                                    with open(type_path+"approved_users.txt", "a", encoding="utf-8") as outfile:
                                        outfile.write(username)
                                        outfile.write("\n")
                                else:
                                    with open(type_path+"rejected_users.txt", "a", encoding="utf-8") as outfile:
                                        outfile.write(username)
                                        outfile.write("\n")



def remove_rejected_data_medium(rejected_filepath, folderpath):
    with open(rejected_filepath, "r", encoding="utf-8") as infile:
        data = infile.readlines()
        file_data = ["messages_" + item.strip() + ".txt" for item in data]
        #print(file_data)
        for msg_file in os.listdir(folderpath):
            #print(msg_file)
            msg_path = os.path.join(folderpath, msg_file)
            try:
                if os.path.isfile(msg_path) and msg_file in file_data:
                    os.unlink(msg_path)
            except Exception as e:
                print(e)
        
def remove_rejected_data_twitter(rejected_filepath, folderpath):
    with open(rejected_filepath, "r", encoding="utf-8") as infile:
        data = infile.readlines()
        file_data = [item.strip() + "_messages.txt" for item in data]
        #print(file_data)
        for msg_file in os.listdir(folderpath):
            #print(msg_file)
            msg_path = os.path.join(folderpath, msg_file)
            try:
                if os.path.isfile(msg_path) and msg_file in file_data:
                    os.unlink(msg_path)
            except Exception as e:
                print(e)

def get_twitter_stats():
    print("TWITTER STATISTICS")
    path = "./twitter/"
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for folder in folders:
        if folder != "old":
            print(folder)
            contents = os.listdir(os.path.join(path,folder))
            print(len(contents) - 1)
            wordcount_country = 0
            for file in contents:
                if file != "metadata_users.txt":
                    #print(file)
                    with open("./twitter/{0}/".format(folder)+file, "r", encoding="utf-8") as infile:
                        data = infile.read().split(" ")
                        wordcount_country += len(data)
            print(wordcount_country)
                

def get_medium_stats():
    print("MEDIUM STATISTICS")
    path = "./medium/"
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for folder in folders:
        print(folder)
        contents = os.listdir(os.path.join(path,folder))
        print(len(contents) - 1)
        wordcount_country = 0 
        for file in contents:
            if file != "metadata_users.txt" and file != "metadata_users_new.txt":
                #print(file)
                with open("./medium/{0}/".format(folder)+file, "r", encoding="utf-8") as infile:
                    data = infile.read().split(" ")
                    wordcount_country += len(data)
        print(wordcount_country)


if __name__ == "__main__":
    consumer_key = 'XXX'
    consumer_secret = 'XXX'

    access_token = 'XXX'
    access_token_secret = 'XXX'

    api = twitter.Api(consumer_key=consumer_key,
      consumer_secret=consumer_secret,
      access_token_key=access_token,
      access_token_secret=access_token_secret, sleep_on_rate_limit=True)
    # Romance, Germanic, Slavic, Indo-Iranian (Hindi and Persian)
    # have a look at: tales from russia, 'from russia with X', 'from the netherlands with youthful italian roots'
    #get_metadata_file_twitter("./twitter/the-netherlands/", "metadata_users.txt")
    #remove_rejected_data_twitter("./twitter/the-netherlands/rejected_users.txt", "./twitter/the-netherlands/")
    #get_twitter_stats()
    #get_medium_stats()
    create_excel_medium("./medium/", "final_metadata_1st_batch.txt")
    #create_excel_twitter("./twitter/", "metadata_users.txt")
    #filter_metadata("./twitter/", "metadata_users.txt", "final_metadata_1st_batch.txt")
