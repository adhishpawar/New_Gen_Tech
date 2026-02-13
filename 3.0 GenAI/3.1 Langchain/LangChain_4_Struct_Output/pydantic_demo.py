from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class Student(BaseModel):
    name: str   = 'Adhish' #Default values
    age: Optional[int] = None  #Optional values and Word with String too
    email: EmailStr        #InBuilt Validation
    cgpa: float = Field(gt=0, lt=10, default= 5, description= "A decimal values representing the CGPA of the student")


new_student = {'name' : "nitish", 'email' : 'adhish@gmail.com', 'cgpa': 5}

student = Student(**new_student)
student_dict = dict(student)
student_json = student.model_dump_json()
print(student_json)