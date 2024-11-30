def handle_topic_weight_management(form_data):
    # Xử lý cho chủ đề Quản lý cân nặng
    variables = {
        "calories": form_data.get("calories"),
        "total_carb": form_data.get("total_carb"),
        "sugar": form_data.get("sugar")
    }
    goal = "Gợi ý món ăn dựa trên nhu cầu calo (giảm cân, tăng cân, duy trì cân nặng)."
    return variables, goal

def handle_topic_heart_health(form_data):
    # Xử lý cho chủ đề Sức khỏe tim mạch
    variables = {
        "cholesterol": form_data.get("cholesterol"),
        "total_fat": form_data.get("total_fat"),
        "sat_fat": form_data.get("sat_fat"),
        "trans_fat": form_data.get("trans_fat")
    }
    goal = "Gợi ý món ăn ít cholesterol và chất béo bão hòa, phù hợp cho người có nguy cơ bệnh tim mạch."
    return variables, goal

def handle_topic_high_protein(form_data):
    # Xử lý cho chủ đề Chế độ ăn giàu protein
    variables = {
        "protein": form_data.get("protein"),
        "calories": form_data.get("calories"),
        "total_fat": form_data.get("total_fat")
    }
    goal = "Gợi ý món ăn giàu protein cho người tập luyện, vận động viên, hoặc người muốn tăng cơ."
    return variables, goal

def handle_topic_sodium_reduction(form_data):
    # Xử lý cho chủ đề Hạn chế sodium
    variables = {
        "sodium": form_data.get("sodium"),
        "total_fat": form_data.get("total_fat"),
        "cholesterol": form_data.get("cholesterol")
    }
    goal = "Đưa ra gợi ý món ăn ít sodium, phù hợp cho người mắc bệnh cao huyết áp."
    return variables, goal

def handle_topic_special_diet(form_data):
    # Xử lý cho chủ đề Chế độ ăn kiêng đặc biệt
    variables = {
        "trans_fat": form_data.get("trans_fat"),
        "total_fat": form_data.get("total_fat"),
        "total_carb": form_data.get("total_carb"),
        "calories": form_data.get("calories")
    }
    goal = "Gợi ý món ăn đáp ứng yêu cầu chế độ ăn đặc biệt như keto (ít carb, nhiều chất béo) hoặc low-fat (ít chất béo)."
    return variables, goal

def generate_prompt(user_data, topic_id):
    # Khởi tạo phần đầu của prompt
    prompt = f"Chào chuyên gia sức khỏe, tư vấn sức khỏe cá nhân, tôi nhờ bạn cho lời khuyên sức khỏe về lựa chọn fastfood, tôi có chủ đề như ở dưới, với các thông số tôi đưa vào và 5 món ăn tôi thích với lượng thông tin về món ăn như tôi sẽ liệt kê, bạn cho tôi lời khuyên về sức khỏe nhé, phân tích về 5 món ăn nữa.\n"
    prompt += f"Tôi đang quan tâm đến {user_data['topic_name']} và tôi được {user_data['goal']}\n"
    # Tùy thuộc vào topic_id mà hiển thị các trường thông tin tương ứng
    prompt += "Đây là thông tin cụ thể hơn về chế độ dinh dưỡng:\n"
    if topic_id == 1:  # Quản lý cân nặng
        prompt += f"Calo: {user_data['form_data']['calories']} kcal\n"
        prompt += f"Carbohydrate: {user_data['form_data']['total_carb']} g\n"
        prompt += f"Đường: {user_data['form_data']['sugar']} g\n"
    elif topic_id == 2:  # Hỗ trợ sức khỏe tim mạch
        prompt += f"Cholesterol: {user_data['form_data']['cholesterol']} mg\n"
        prompt += f"Tổng chất béo: {user_data['form_data']['total_fat']} g\n"
        prompt += f"Chất béo bão hòa: {user_data['form_data']['sat_fat']} g\n"
        prompt += f"Chất béo chuyển hóa: {user_data['form_data']['trans_fat']} g\n"
    elif topic_id == 3:  # Chế độ ăn giàu protein
        prompt += f"Protein: {user_data['form_data']['protein']} g\n"
        prompt += f"Calo: {user_data['form_data']['calories']} kcal\n"
        prompt += f"Tổng chất béo: {user_data['form_data']['total_fat']} g\n"
    elif topic_id == 4:  # Hạn chế sodium
        prompt += f"Sodium: {user_data['form_data']['sodium']} mg\n"
        prompt += f"Tổng chất béo: {user_data['form_data']['total_fat']} g\n"
        prompt += f"Cholesterol: {user_data['form_data']['cholesterol']} mg\n"
    elif topic_id == 5:  # Chế độ ăn kiêng đặc biệt
        prompt += f"Chất béo chuyển hóa: {user_data['form_data']['trans_fat']} g\n"
        prompt += f"Tổng chất béo: {user_data['form_data']['total_fat']} g\n"
        prompt += f"Carbohydrate: {user_data['form_data']['total_carb']} g\n"
        prompt += f"Calo: {user_data['form_data']['calories']} kcal\n"
    prompt += f"Khi đưa ra lời khuyên, nhớ nhắc lại về các thông tin về dinh dưỡng được cung cấp trên nhé. Không yêu cầu tôi cung cấp thêm gì, hãy xưng hô khi đưa ra là Hệ thống\n"
    prompt += f"Ví dụ: Hệ thống khuyên bạn..., bên cạnh đó, sử dụng từ ngữ khẳng định hơn, thay vì Nhắc lại mục tiêu hay gì đó thì Mục tiêu bạn đưa ra là:... (Ngữ cảnh bạn là nhân viên tư vấn)\n"
    prompt += "\nVà đây là 5 món ăn và hàm lượng dinh dưỡng của nó được xếp theo độ ưu tiên từ cao đến thấp, hãy phân tích kỹ từng món ăn và lý do lựa chọn nhé:\n"
    
    # Gắn các món ăn được khuyến nghị với thông tin chi tiết
    for i, meal in enumerate(user_data['meals']):
        prompt += f"{i+1}. Món {meal['name']} - Calories: {meal['calories']} kcal, Protein: {meal['protein']} g, "
        prompt += f"Total Fat: {meal['total_fat']} g, Sat Fat: {meal['sat_fat']} g, Trans Fat: {meal['trans_fat']} g, "
        prompt += f"Cholesterol: {meal['cholesterol']} mg, Sodium: {meal['sodium']} mg, "
        prompt += f"Total Carb: {meal['total_carb']} g, Sugar: {meal['sugar']} g\n"
    
    prompt += "\nHãy đưa ra lời khuyên bổ ích cho tôi về dinh dưỡng và món ăn nhé, nhớ nhắc lại những gì tôi đã cung cấp và nhận xét về những thông tin tôi đã đưa ra, viết thật ngắn gọn thôi.\n"

    return prompt

