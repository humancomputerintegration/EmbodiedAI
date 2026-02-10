def get_recognition_prompt(image, task, handedness, hands_obj, user_profile, cached_output):

    return [
        {
            "role": "developer",
            "content": f"""
            You will be provided:
            “the handedness that the user is holding or touching the object”: {handedness} and {hands_obj}. Use this information to decide which hand/limb to use in the movement instructions. Account for user profile: {user_profile} to determine which limb and handedness preference.
            If the object in the image is related to the user's goal define by {task}, use the provided handedness for movement instructions. If the object in the image is unrelated to the user's goal, use the other hand.
            If the handedness is unclear, or both hands will work, refer to customized preference in the user profile to determine the handedness.
            You will:
            1. Analyze the provided image where hands are interacting with an object.
            2. Identify the object being held or manipulated by the hands.
            3. Based on the user's goal defined by '{task}', identify the mechanism to help achieve {task}:
            - ask yourself: "how do I use this object to acheive {task}?"
            - take into consideration if the object has a special feature that is uncommon to this context
            - only output the mechanism that is most relevant to the user's goal {task}
            4. When reasoning about the mechanism and use of the object, always consider if there are standard preparation or safety steps required before the main action. Explicitly mention these steps if they are necessary for the object to function as intended.
            5. If location information is available in the user profile '{user_profile}', use it for your reasoning to determine the object/mechanism and functionality of the object.
            6. Explain the object's functionality in the current context, detailing how it is typically used aligned with the user's goal defined by '{task}'. Consider the current state of the object and ignore irrelevant steps or steps that are already completed.
            7. Rule out the functionality or mechanism that does not work with prior failed reasoning result: '{cached_output}' (see LLM output (recognition)). Do not repeat the prior failed reasoning result. Try another mechanism or functionality.
            8. If the object has multiple mechanisms or functions, propose only one that makes the most sense in this context. Do not list all the possible functions or optional functions.
            9. If the user's hands do not interact with a specific object in the image, use the contextual information from the image and the user's request (task) to provide relevant information.
            9. If you suggest any target angles for object manipulation, look at the current image and figure out its current angle, recalculate and update the angles by simulating the rotation you suggested and ensure they correspond to the object's current state and common sense, then document these angles clearly.
            10. Review your output by:
            - making sure that you only provided one mechanism
            - the description aligns with the user's goal "{task}" and does not provide any extra non-essential information
            - all the angles are consistent with the object's current state and end goal
            """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "User Task: " + task + "\n" + "User Profile: " + str(user_profile) + "\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            ]
        }
    ]

def get_movements_prompt(recognition_response, limb_directions, user_profile, cached_output="", task=None): 
    return [
        {
            "role": "developer", 
            "content": 
            f'''
            You will be provided:
            "descriptions of the object and its mechanism": {str(recognition_response)}. Use this information to formulate the movement instructions.
            "current body position described by where the limbs are pointing. Note that the wrist is referring to the thumb to point": {limb_directions}
            "Prior failed reasoning result": {cached_output}

            <instructions>
                You are a physical assistant that can control the user's body. 
                Your task is to analyze the descriptions of the object and its mechanism, and the user's goal to generate step-by-step movement instructions to interact with the object to achieve their goal. 
                Follow these steps to complete the task:
                1. Use the current body position described by {limb_directions} to determine the starting position of each limb. Avoid redundant movements by considering the current orientation and position of the limbs. For example:
                - If the limb is already in the required position, do not mention this steps.
                - If a movement is already completed or unnecessary, do not include it in the output."
                - If the limb is pointing in the opposite direction, include a step to reorient it before proceeding with the task.
                - If a limb is already maintaining grip or holding the object or a position, do not mention the steps to reach or grip or hold the object. 
                2. Only include the minimal set of movements required to achieve the user's goal. Do not add extra steps such as holding or gripping objects unless they are strictly necessary for the action
                3. Rule out the functionality or mechanism that does not work with prior failed reasoning result: {cached_output} (see LLM output (movement):). Do NOT repeat the prior failed reasoning result. Try another mechanism or functionality. 
                4. Ensure your instructions are easy to follow and consider the user's perspective.
                5. Ensure each movement instruction is an actionable movement instruction that is directly related to the user's goal. Remove all the adjectives that are not critical.
                6. Be specific and reply in a formatted list as follows: 
                - 'limb:' followed by one word that details which limb of the body the user is using to perform the task (e.g., foot or hand).
                - 'handedness': followed by one word that details on which side of the body this limb is, with respect to the person's point of view in the photo (e.g., left or right). 
                - 'movement': start with handness like "Left or Right hand: ..." and then describe in detail how the limb is moving step by step. (e.g., firstly, which joint, flexion, or extension, and secondly ...) Include the required angle of the movement to complete each step (e.g., turn the object for how many degrees.)
                If it's a sequence of movements, list all the required movements in sequence; if it's a bimanual or bipedal task, output movements for each hand or each foot. 
                7. Ensure your output contains no XML tags or additional formatting; it should be a plain text response.
                8. Carefully consider the angle of the movement and the direction of the movement by explicitly stating the angle and direction of the movement if it is essential to achieve the goal. 
                9. Review the angles and ensure the instructions match the current user limb direction, object state, and common sense. Recalculate and update the angles by simulating the rotation you suggested. 
                10. Keep instructions simple and concise. Avoid unnecessary complexity.
                11. If you cannot determine the movement or mapping, skip the step.
                12. Before outputting, review your instructions to ensure:
                - the provided instructions are solving for the user's task: "{task}".
                - remove any instructions that do not directly change the state required for the user's goal.
            </instructions>

            <examples>
                Example1:
                <input>
                    Current body pose: Left hand is holding a tea bowl with a matcha whisk in it. Right hand is free.
                    Task: teach the user how to whisk matcha using the matcha whisk (chasen) with a zigzag motion.
                </input>
                <output>
                    -limb: hand
                    -handedness: right
                    -movement:
                        1) Right hand: reach the whisk.
                        2) Right hand: grip the whisk.
                        3) Right hand: start zigzag motions to whisk the matcha.
                </output>

                Example2 (an example of a bimanual task):
                <input>
                    Current body pose: Left foot is next to a bike pump. Both hands are resting on the handle of the bike pump.
                    Task: teach the user how to use the bike pump to inflate the tires.
                </input>
                <output>
                    limb: foot
                    handedness: left
                    movement:
                        1) Left foot: twist your ankle to move your left foot onto the pedal of the bike pump.
                    limb: arm
                    handedness: left and right
                    movement:
                        2-1) Left arm: pull the handle up.
                        2-2) Right arm: pull the handle up.
                    limb: arm
                    handedness: left and right
                    movement:
                        3-1) Left arm: push the handle down.
                        3-2) Right arm: push the handle down.
            </output>

            '''
        },
        {
            "role": "user",
            "content": [
                {  
                "type": "text",
                "text": "User Profile: " + str(user_profile) + "\n"
                },

            ],
        }
    ]


def get_gestures_oneshot_prompt(movement_response, joint_list, limb_directions):
    return [
        {
            "role": "developer", 
            "content": 
            f'''
            You will be provided:
            "a movement for operating an object or achieving a goal": {movement_response}"
            "a joint_list (the movements of the joints, and the direction that the joints can turn relative to the neutral pose)”: {joint_list}
            "the neutral pose”: it is a person standing upright with the arms resting beside the thighs. Use this neutral pose as a baseline (0 degrees) to calculate the direction and angle of the movements.
            "current body position described by where the limbs are pointing. Note that the wrist is referring to the thumb to point": {limb_directions}
            
            <instructions>
                You are tasked with translating a list of movement instructions into specific joint movements using a provided joint list. Follow these steps to complete the task:

                1. Carefully read the provided movement instructions, which include details about the limb, handedness, and specific actions to be performed.
                2. For each movement, use the joint_list: {joint_list} to describe the the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] Ensure that each movement is directly linked to the action it corresponds to. 
                3. Consider the current body position described by {limb_directions} to determine which body part should move in which direction.
                4. Be as efficient and succinct as possible. If a movement instruction can be achived with one muscle movement, use one movement, instead of multiple joint movements.
                5. Ensure you only use the [joint][movement][direction] pairs in the joint_list.
                6. Structure your output for each movement in the following format: <handedness>[joint][movement][direction][angle]. Make sure to include all have parameters. If more than one joint movement is required, use the following format: <handedness>[joint][movement][direction][angle] \n <handedness>[joint][movement][direction][angle]. 
                7. Ensure that your output does not contain any XML tags but keep the "<left>" or "<right>" tag to indicate the handedness of the movement and should not be modified.
                8. Provide a clear separation between different limbs and movements in your output for better readability.
                9. Review your output to ensure that you only choose the joint movement pairs ([joint][movement][direction]) from the joint_list. Also ensure accuracy and completeness before finalizing it.
                10. If you cannot determine the movement or mapping, output <skip>.
                11. Before outputting, review your instructions to ensure:
                - this exact instruction format is respected, including brackets, with exactly 5 elements: <handedness>[joint][movement][direction][angle]
            </instructions> 

            <example>
                <input>
                    -limb: hand
                    -handedness: right
                    -movement:
                        1) Right hand: reach the whisk.
                        2) Right hand: grip the whisk.
                        3) Right hand: start zigzag motions to whisk the matcha.
                </input>
                <output>
                    1) Right hand: reach the whisk.  
                    <right>[shoulder][flexion][upward][90]
                    <right>[elbow][flexion][inward][90]
                    2) Right hand: grip the whisk.  
                    <right>[index][flexion][inward][5]
                    <right>[middle][flexion][inward][5]
                    <right>[thumb][flexion][inward][5]
                    3) Right hand: start zigzag motion to whisk the matcha. 
                    <right>[wrist][extension][upward][10]
                    <right>[wrist][flexion][downward][10]
                </output>
            </example>

            <example>
                <input>
                limb: foot
                handedness: left
                movement:
                    1) Left foot: twist your ankle to move your left foot onto the pedal of the bike pump.
                limb: arm
                handedness: left and right
                movement:
                    2-1) Left arm: pull the the handle up.
                    2-2) Right arm: pull the handle up.
                limb: arm
                handedness: left and right
                movement:
                    3-1) Left arm: push the the handle down.
                    3-2) Right arm: push the handle down.
                </input>
                <output>
                    1) Left foot: twist your ankle to move your left foot onto the pedal of the bike pump. 
                        <left>[ankle][inversion][inward][10]
                    2-1) Left arm: pull the the handle up.
                    2-2) Right arm: pull the handle up.
                    <left>[elbow][flexion][upward][50]
                    <right>[elbow][flexion][upward][50]
                    3-1) Left arm: push the the handle down.
                    3-2) Right arm: push the handle down.
                    <left>[elbow][extension][downward][50]
                    <right>[elbow][extension][downward][50]
                </output>
            </example>
            '''
        },
        {
            "role": "user",
            "content": [
                {  
                "type": "text",
                "text": 
                f'''
                Describe the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][the direction of where the limb is moving towards to in relationship to the center of the body][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] to achieve the following movements {movement_response}. 
                If it's a bimanual or bipedal task, then output movement description for each hand or each foot.
                '''
                },
            ],
        }
    ]

# with checkpoints
def get_gestures_prompt(checkpoints_response, joint_list, limb_directions):
    return [
        {
            "role": "developer", 
            "content": 
            f'''
            You will be provided:
            "a movement for operating an object or achieving a goal": {checkpoints_response}
            "a joint_list (the movements of the joints, and the direction that the joints can turn relative to the neutral pose)”: {joint_list}
            "the neutral pose”: it is a person standing upright with the arms resting beside the thighs. Use this neutral pose as a baseline (0 degrees) to calculate the direction and angle of the movements.
            "current body position described by where the limbs are pointing. Note that the wrist is referring to the thumb to point": {limb_directions}

            <instructions>
                You are tasked with translating a list of movement instructions into specific muscle movements using a provided joint list. Follow these steps to complete the task:

                1. Carefully read the provided movement instructions, which include details about the limb, handedness, and specific actions to be performed.
                        2. For each movement, use the joint_list: {joint_list} to describe the the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] Ensure that each movement is directly linked to the action it corresponds to. 
                        3. Consider the current body position described by {limb_directions} to determine which body part should move in which direction.
                        4. Be as efficient and succinct as possible. If a movement instruction can be achived with one muscle movement, use one movement, instead of multiple joint movements.
                        5. Ensure you only use the [joint][movement][direction] pairs in the joint_list.
                        6. Structure your output for each movement in the following format: <handedness>[joint][movement][direction][angle]. Make sure to include all have parameters. If more than one joint movement is required, use the following format: <handedness>[joint][movement][direction][angle] \n <handedness>[joint][movement][direction][angle]. 
                        7. Ensure that your output does not contain any XML tags but keep the "<left>" or "<right>" tag to indicate the handedness of the movement and should not be modified.
                        8. Provide a clear separation between different limbs and movements in your output for better readability.
                        9. Review your output to ensure that you only choose the joint movement pairs ([joint][movement][direction]) from the joint_list. Also ensure accuracy and completeness before finalizing it.
                        10. If you cannot determine the movement or mapping, output <skip>.
                        11. Before outputting, review your instructions to ensure:
                        - follow this exact instruction format <limb>[handedness][movement joint][direction][angle]
            </instructions> 

            <examples>
                <example>
                    <input>
                        1) Right hand: reach the whisk.  
                    </input>
                    <output>
                        <right>[shoulder][flexion][upward][90]
                        <right>[elbow][flexion][inward][90]
                    </output>
                </example>
                
                <example>
                    <input>
                        2) Right hand: grip the whisk.  
                    </input>
                    <output>
                        <right>[index][flexion][inward][5]
                        <right>[middle][flexion][inward][5]
                        <right>[thumb][flexion][inward][5]
                    </output>
                </example>

                <example>
                    <input>
                        3) Right hand: start zigzag motion to whisk the matcha. 
                    </input>
                    <output>
                        <right>[wrist][extension][upward][10]
                        <right>[wrist][flexion][downward][10]
                    </output>
                </example>

                <example>
                    <input>
                        1) Left foot: twist your ankle to move your left foot onto the pedal of the bike pump. 
                    </input>
                    <output>
                        <left>[ankle][inversion][inward][10]
                    </output>
                </example>

                <example>
                    <input>
                        2-1) Left arm: pull the the handle up.
                        2-2) Right arm: pull the handle up. 
                    </input>
                    <output>
                        <left>[elbow][flexion][upward][50]
                        <right>[elbow][flexion][upward][50]
                    </output>
                </example>
            </examples>

            '''
        },
        {
            "role": "user",
            "content": [
                {  
                "type": "text",
                "text": 
                f'''
                Describe the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] to achieve the following movements {checkpoints_response}. 
                If it's a bimanual or bipedal task, then output movement description for each hand or each foot.

                '''
                },
            ],
        }
    ]

def get_checkpoints_prompts(image, task, movements_response, handedness):
    return [
        {
            "role": "developer", 
            "content": 
            f'''
                <instructions>
                You will receive an image depicting the user's current state, a description of the user's task, and a sequence of movements the user needs to make. 
                Your objective is to analyze the provided information and choose the next step from the movements list with the handedness. 

                Follow these steps to complete the task:
                1. Examine the image to understand the user's current state and context.
                2. Follow the given handedness information: {handedness} to determine Left hand or Right hand. 
                3. Read the description of the user's task to understand what they are trying to achieve.
                4. Review the sequence of movements to identify what actions the user has already taken.
                5. Based on your analysis, choose the next step from the movements list for the user to continue towards their goal.
                6. If you determine that the user has already achieved their goal, simply output "done".
                7. Ensure that your output contains no XML tags or additional formatting; it should be a plain text response.
            </instructions>
            <example>
                    <input>
                        Image: Left hand is holding a tea bowl. Right hand is holding a whisk.
                        Task: teach the user how to whisk matcha using the matcha whisk (chasen) with a zigzag motion.
                        -limb: hand
                        -handedness: right
                        -movement:
                            1) Right hand: reach the whisk.
                            2) Right hand: grip the whisk.
                            3) Right hand: start zigzag motions to whisk the matcha.
                    </input>
                    <output>
                        3) Right hand: start zigzag motions to whisk the matcha.
                    </output>
            </example>
            '''
        },
        {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{image}"
                    } 
                },
                {  
                "type": "text",
                "text": 
                f'''
                Task: {task}
                Movements: {movements_response}
                '''
                }
            ],
        },
    ]
