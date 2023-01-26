#Object–relational mapping (ORM, O/RM, and O/R mapping tool)
#if a package is not found, install it using pip
#pip install sqlalchemy
#pip install uuid #for generating unique id

from sqlalchemy import create_engine, ForeignKey, String, Integer, Column
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.orm import sessionmaker
import uuid

Base = declarative_base() 
