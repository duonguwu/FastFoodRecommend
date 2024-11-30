import pandas as pd
import numpy as np
import os
import google.generativeai as genai

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.stats import norm  
from typing import List, Optional, Dict
from dotenv import load_dotenv

from prompt_generator import * 

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")
      
app = FastAPI(title="Food Recommendation System API")

# Mở CORS cho phép tất cả domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],  # Cho phép tất cả các header
)

# Các endpoint của bạn ở đây
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Load data
df = pd.read_csv("fastfood_postprocess.csv")

class BayesianRecommender:
    def __init__(self, data, variables):
        self.data = data
        self.variables = variables
        self.data_normalized = self._normalize_data()
        self.params = self._calculate_distribution_params()
    
    def _normalize_data(self):
        normalized = self.data.copy()
        for col in self.variables:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            normalized[col] = self.data[col].fillna(0)
            normalized[col] = (self.data[col] - min_val) / (max_val - min_val)
        return normalized
    
    def _calculate_distribution_params(self):
        params = {}
        for col in self.variables:
            params[col] = {
                'mean': self.data_normalized[col].mean(),
                'std': self.data_normalized[col].std()
            }
        return params
    
    def _calculate_likelihood(self, row, target_values):
        # Chuẩn hóa giá trị mục tiêu
        target_normalized = {}
        for var in self.variables:
            target_normalized[var] = (target_values[var] - self.data[var].min()) / \
                                   (self.data[var].max() - self.data[var].min())
        
        # Tính likelihood cho mỗi biến
        likelihoods = []
        for var in self.variables:
            likelihood = norm.pdf(row[var], 
                                target_normalized[var], 
                                self.params[var]['std'])
            likelihoods.append(likelihood)
        
        # Kết hợp các likelihood theo mạng Bayes
        return np.prod(likelihoods)
    
    def recommend(self, target_values, restaurants=None, top_n=5):
        # Tính likelihood cho tất cả các món
        self.data_normalized['likelihood'] = self.data_normalized.apply(
            lambda row: self._calculate_likelihood(row, target_values),
            axis=1
        )
        
        # Lọc theo nhà hàng nếu có
        filtered_data = self.data
        if restaurants:
            filtered_data = self.data[self.data['restaurant'].isin(restaurants)]
        
        # Sắp xếp và lấy top n món có likelihood cao nhất
        top_items = self.data_normalized.loc[filtered_data.index].sort_values('likelihood', ascending=False).head(top_n)
        
        recommendations = []
        for idx, row in top_items.iterrows():
            original_row = self.data.loc[idx]
            recommendations.append({
                'item': original_row['item'],
                'restaurant': original_row['restaurant'],
                'nutritional_info': {var: original_row[var] for var in self.variables},
                'match_score': row['likelihood']
            })
        
        return recommendations

# Định nghĩa các models cho API
class WeightManagementForm(BaseModel):
    calories: float = Field(..., description="Target calories value")
    total_carb: float = Field(..., description="Target carbohydrates value")
    sugar: float = Field(..., description="Target sugar value")

class HeartHealthForm(BaseModel):
    cholesterol: float = Field(..., description="Target cholesterol value")
    total_fat: float = Field(..., description="Target total fat value")
    sat_fat: float = Field(..., description="Target saturated fat value")
    trans_fat: float = Field(..., description="Target trans fat value")

class ProteinDietForm(BaseModel):
    protein: float = Field(..., description="Target protein value")
    calories: float = Field(..., description="Target calories value")
    total_fat: float = Field(..., description="Target total fat value")

class LowSodiumForm(BaseModel):
    sodium: float = Field(..., description="Target sodium value")
    total_fat: float = Field(..., description="Target total fat value")
    cholesterol: float = Field(..., description="Target cholesterol value")

class SpecialDietForm(BaseModel):
    trans_fat: float = Field(..., description="Target trans fat value")
    total_fat: float = Field(..., description="Target total fat value")
    total_carb: float = Field(..., description="Target carbohydrates value")
    calories: float = Field(..., description="Target calories value")

class RecommendationRequest(BaseModel):
    topic_id: int = Field(..., description="Topic ID (1-5)")
    restaurants: Optional[List[str]] = Field(None, description="List of selected restaurants (max 3)")
    form_data: Dict = Field(..., description="Form data based on selected topic")

class NutritionalInfo(BaseModel):
    calories: float = 0.0
    total_fat: float = 0.0
    sat_fat: float = 0.0
    trans_fat: float = 0.0
    cholesterol: float = 0.0
    sodium: float = 0.0
    total_carb: float = 0.0
    sugar: float = 0.0
    protein: float = 0.0



class FoodRecommendation(BaseModel):
    item: str
    restaurant: str
    nutritional_info: NutritionalInfo
    match_score: float

class RecommendationResponseAI(BaseModel):
    topic_name: str
    goal: str
    recommendation_text: str

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    recommendations: List[FoodRecommendation]

# Khởi tạo các recommender cho từng chủ đề
TOPIC_RECOMMENDERS = {
    1: BayesianRecommender(df, ['calories', 'total_carb', 'sugar']),
    2: BayesianRecommender(df, ['cholesterol', 'total_fat', 'sat_fat', 'trans_fat']),
    3: BayesianRecommender(df, ['protein', 'calories', 'total_fat']),
    4: BayesianRecommender(df, ['sodium', 'total_fat', 'cholesterol']),
    5: BayesianRecommender(df, ['trans_fat', 'total_fat', 'total_carb', 'calories'])
}

def process_recommendations(topic_id: int, form_data: Dict, restaurants: Optional[List[str]] = None):
    recommender = TOPIC_RECOMMENDERS.get(topic_id)
    if not recommender:
        raise HTTPException(status_code=400, detail="Invalid topic ID")
    
    recommendations = recommender.recommend(
        target_values=form_data,
        restaurants=restaurants,
        top_n=5
    )
    
    # Lấy ra danh sách các món ăn từ khuyến nghị
    recommended_items = [rec['item'] for rec in recommendations]
    
    # Đọc dữ liệu từ file CSV (giả sử bạn có DataFrame chứa dữ liệu món ăn)
    df = pd.read_csv("fastfood_postprocess.csv")
    
    # Lọc các món ăn khuyến nghị từ DataFrame
    recommended_data = df[df['item'].isin(recommended_items)]
    
    processed_recommendations = []
    
    # Duyệt qua các món ăn khuyến nghị và lấy thông tin dinh dưỡng
    for rec in recommendations:
        # Lọc thông tin dinh dưỡng của món ăn trong recommended_data
        item_data = recommended_data[recommended_data['item'] == rec['item']].iloc[0]
        
        # Đảm bảo không có giá trị None, sử dụng giá trị trong item_data từ DataFrame
        nutritional_info = {
            'calories': item_data['calories'] if not pd.isnull(item_data['calories']) else 0.0,
            'total_fat': item_data['total_fat'] if not pd.isnull(item_data['total_fat']) else 0.0,
            'sat_fat': item_data['sat_fat'] if not pd.isnull(item_data['sat_fat']) else 0.0,
            'trans_fat': item_data['trans_fat'] if not pd.isnull(item_data['trans_fat']) else 0.0,
            'cholesterol': item_data['cholesterol'] if not pd.isnull(item_data['cholesterol']) else 0.0,
            'sodium': item_data['sodium'] if not pd.isnull(item_data['sodium']) else 0.0,
            'total_carb': item_data['total_carb'] if not pd.isnull(item_data['total_carb']) else 0.0,
            'sugar': item_data['sugar'] if not pd.isnull(item_data['sugar']) else 0.0,
            'protein': item_data['protein'] if not pd.isnull(item_data['protein']) else 0.0
        }
        
        print("nutritional_info", nutritional_info)
        
        processed_recommendations.append(
            FoodRecommendation(
                item=rec['item'],
                restaurant=rec['restaurant'],
                nutritional_info=NutritionalInfo(**nutritional_info),
                match_score=float(rec['match_score'])
            )
        )
    
    return processed_recommendations

prompt = ""

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        # Validate topic_id
        if request.topic_id not in range(1, 6):
            raise HTTPException(status_code=400, detail="Invalid topic ID")
            
        # Validate restaurants (if provided)
        if request.restaurants and len(request.restaurants) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 restaurants allowed")
            
        # Process recommendations based on topic
        recommendations = process_recommendations(
            topic_id=request.topic_id,
            form_data=request.form_data,
            restaurants=request.restaurants
        )
        
        return RecommendationResponse(
            success=True,
            message="Recommendations generated successfully",
            recommendations=recommendations
        )
        
    except Exception as e:
        return RecommendationResponse(
            success=False,
            message=str(e),
            recommendations=[]
        )

@app.post("/api/ai_recommend", response_model=RecommendationResponseAI)
async def get_recommendations_ai(request: RecommendationRequest):
    try:
        # Validate topic_id
        if request.topic_id not in range(1, 6):
            raise HTTPException(status_code=400, detail="Invalid topic ID")
            
        # Validate restaurants (if provided)
        if request.restaurants and len(request.restaurants) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 restaurants allowed")
            
        # Process recommendations based on topic
        recommendations = process_recommendations(
            topic_id=request.topic_id,
            form_data=request.form_data,
            restaurants=request.restaurants
        )

        # Xác định tên chủ đề và mục tiêu tương ứng
        if request.topic_id == 1:
            topic_name = "Quản lý cân nặng"
            variables, goal = handle_topic_weight_management(request.form_data)
        elif request.topic_id == 2:
            topic_name = "Sức khỏe tim mạch"
            variables, goal = handle_topic_heart_health(request.form_data)
        elif request.topic_id == 3:
            topic_name = "Chế độ ăn giàu protein"
            variables, goal = handle_topic_high_protein(request.form_data)
        elif request.topic_id == 4:
            topic_name = "Hạn chế sodium"
            variables, goal = handle_topic_sodium_reduction(request.form_data)
        elif request.topic_id == 5:
            topic_name = "Chế độ ăn kiêng đặc biệt"
            variables, goal = handle_topic_special_diet(request.form_data)
        else:
            topic_name = "Không xác định" 

        # Tạo dữ liệu người dùng
        user_data = {
            'topic_name': topic_name,  
            'form_data': request.form_data,
            'goal': goal,
            'variables': variables,  
            'meals': [
                {
                    'name': rec.item,
                    'calories': rec.nutritional_info.calories,
                    'total_fat': rec.nutritional_info.total_fat,
                    'sat_fat': rec.nutritional_info.sat_fat,
                    'trans_fat': rec.nutritional_info.trans_fat,
                    'cholesterol': rec.nutritional_info.cholesterol,
                    'sodium': rec.nutritional_info.sodium,
                    'total_carb': rec.nutritional_info.total_carb,
                    'sugar': rec.nutritional_info.sugar,
                    'protein': rec.nutritional_info.protein
                } for rec in recommendations
            ]
        }
        
        # Tạo prompt cho người dùng
        prompt = generate_prompt(user_data, request.topic_id)
        print("Generated Prompt: ", prompt)

        ai_response = model.generate_content(prompt)
        print("AI Response: ", ai_response)

        # Trả về kết quả với format mới
        return RecommendationResponseAI(
            topic_name=user_data['topic_name'],
            goal=goal,
            recommendation_text=ai_response.text
        )
        
    except Exception as e:
        return RecommendationResponse(
            success=False,
            message=str(e),
            recommendations=[]
        )

@app.get("/api/topic-form/{topic_id}")
async def get_topic_form(topic_id: int):
    form_schemas = {
        1: {
            "title": "Weight Management",
            "fields": {
                "calories": {"type": "number", "required": True, "min": 0},
                "total_carb": {"type": "number", "required": True, "min": 0},
                "sugar": {"type": "number", "required": True, "min": 0}
            }
        },
        2: {
            "title": "Heart Health",
            "fields": {
                "cholesterol": {"type": "number", "required": True, "min": 0},
                "total_fat": {"type": "number", "required": True, "min": 0},
                "sat_fat": {"type": "number", "required": True, "min": 0},
                "trans_fat": {"type": "number", "required": True, "min": 0}
            }
        },
        # Thêm các chủ đề còn lại...
    }
    
    if topic_id not in form_schemas:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return form_schemas[topic_id]
