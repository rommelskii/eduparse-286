from pymongo import MongoClient

class Database:
    def __init__(self, database_url, collection_name):
        self.client = MongoClient(database_url)
        self.db = self.client[collection_name]
        self.collection = self.db["users"]

    def addUser(self, name):
        if self.fetchUser(name) is not None:
            return None
        self.collection.insert_one({
            "user_name": name,
            "sessions": []
        })
    def deleteUser(self, name):
        if self.fetchUser(name) is None:
            print("ERR: user not found")
            return
        self.collection.delete_one({ "user_name": name })

    def fetchUser(self, name):
        user = self.collection.find_one({ "user_name": name })
        return user

    def fetchUserSessions(self, name):
        sessions = self.fetchUser(name)['sessions']
        return sessions

    def fetchSession(self, name, session_name):
        sessions = self.fetchUserSessions(name)
        result = next((item for item in sessions if item.get("session_name") == session_name), None)
        if result is not None:
            return result
        else:
            print("ERR: cannot find the session")
            return

    def addSession(self, name, session):
        sessions = self.fetchUserSessions(name)
        sessions.append(session)
        new_fields = { "sessions": sessions }
        self.collection.update_one( {"user_name": name}, {"$set": new_fields} )

    def removeSession(self, name, session_name):
        sessions = self.fetchUserSessions(name)
        if len(sessions) == 0:
            print("ERR: session empty")
            return
        result = next((item for item in sessions if item.get("session_name") == session_name), None)
        if result is not None:
            sessions.remove(result)
        else:
            print("ERR: cannot find the session")
            return
        new_fields = { "sessions": sessions }
        self.collection.update_one( {"user_name": name}, {"$set": new_fields} )
     
     
