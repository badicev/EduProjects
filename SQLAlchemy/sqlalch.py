#Object–relational mapping (ORM, O/RM, and O/R mapping tool)
#if a package is not found, install it using pip
#pip install sqlalchemy
#pip install uuid #for generating unique id

from sqlalchemy import create_engine, ForeignKey, String, Integer, Column
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.orm import sessionmaker
import uuid

Base = declarative_base() #returns a new base class from which all mapped classes should inherit

def generate_uuid():
    return str(uuid.uuid4())


class users(Base):
    __tablename__ = "users"
    userID = Column("userID", String, primary_key = True, default= generate_uuid)
    firstName = Column("firstName", String)
    profileName = Column("profileName", String)
    email = Column ("email", String)
    
    def __init__(self, firstName, lastName, profileName, email):
        self.firstName = firstName
        self.lastName = lastName
        self.profileName = profileName
        self.email = email
        
class posts(Base):
    __tablename__ = "posts"
    postId = Column("postId", String, primary_key=True, default=generate_uuid)
    userId = Column("userId", String, ForeignKey("users.userID"))  
    postContent = Column("postContent", String)
    
    def __init__(self, userId, postContent):
        self.userId = userId
        self.postContent = postContent
        
class likes(Base):
    __tablename__ = "likes"
    likeId = Column("likeId", String, primary_key=True, default = generate_uuid)
    userID = Column("userID", String, ForeignKey("users.userID"))  
    postId = Column("postId", String, ForeignKey("posts.postId")) 
    
    def __init__(self, userId, postId):
        self.userId = userId
        self.postId = postId
    
        
def addUser(firstName, lastName, profileName, email, session):
    exist = session.query(users).filter(users.email==email).all()
    if len(exist)>0:
         print("E-mail adress already exist!")
    else:
        user = users(firstName, lastName, profileName, email) 
        session.add(user) #staging era
        session.commit() #apply your changes
        print("user added to db")
    
def addPost(userId, postContent,session):
    newPost = posts(userId, postContent)
    session.add(newPost)    
    session.commit()
    
def addLike(userId, postId):
    like = likes(userId, postId)
    session.add(like)
    session.commit()
    print("Like added to db. ")
    
      
db = "sqlite:///SQLAlchemy/socialDB.db"
engine = create_engine(db)
Base.metadata.create_all(bind=engine)

Session = sessionmaker(bind=engine)
session = Session()

 #Create a user
firstName = "Bat"
lastName = "Hunter"
profileName = "Eba1574"
email = "bathunt@gmail.com"
addUser(firstName, lastName, profileName, email, session)


#Create a post
userId = "0e7841dd-31a2-400f-b007-824e0b9b9afe"
postId = "11d238e7-1219-40d6-9b59-4e07ca02f1c8"
postContent = "This is my (third user) second post."
#addPost(userId, postContent, session)


allPosts = session.query(posts).filter(posts.userId==userId)
postsFilteredByUser = [post.postContent for post in allPosts]
print(postsFilteredByUser)

#Like a post
#addLike(userId, postsId)


postLikes = session.query(likes).filter(likes.postId==postId).all()
print(len(postLikes))


#join | use 2 tables
usersLikedPost = session.query(users, likes).filter(likes.postId==postId).filter(likes.userID==users.userID)

for u in usersLikedPost:
    print(u["users"].firstName, u["users".lastName])