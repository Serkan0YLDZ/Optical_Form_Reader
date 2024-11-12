import cv2
import numpy as np
from typing import List, Tuple, Dict
import string


class OpticalFormReader:
    def __init__(self, y_threshold: int = 10, filling_threshold: float = 0.6):
        self.y_threshold = y_threshold
        self.filling_threshold = filling_threshold

    def detect_circles_and_marks(self, image_path: str, answer_key: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        image = cv2.imread(image_path)
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        detected_circles = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            grouped_circles = self.group_circles(
                [{
                    'x': int(c[0]),
                    'y': int(c[1]),
                    'radius': int(c[2]),
                    'is_filled': cv2.mean(gray, mask=self.create_circle_mask(gray, int(c[0]), int(c[1]), int(c[2])-2))[0] < 128
                } for c in circles[0, :]]
            )

            for group_idx, group in enumerate(grouped_circles):
                if group_idx >= len(answer_key):
                    break

                correct_answer = answer_key[group_idx]
                correct_index = string.ascii_lowercase.index(correct_answer)

                marked_circles = [c for c in group if c['is_filled']]
                is_marked = len(marked_circles) == 1

                correct_circle = group[correct_index]
                overlay = output.copy()
                cv2.circle(
                    overlay,
                    (correct_circle['x'], correct_circle['y']),
                    correct_circle['radius']-2,
                    (0, 255, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

                for idx, circle in enumerate(group):
                    x, y, r = circle['x'], circle['y'], circle['radius']
                    is_filled = circle['is_filled']

                    cv2.circle(output, (x, y), r, (0, 0, 0), 2)

                    if is_marked and is_filled:
                        if idx == correct_index:
                            cv2.circle(output, (x, y), r-2, (0, 255, 0), -1)
                        else:
                            cv2.circle(output, (x, y), r-2, (0, 0, 255), -1)

                    circle['group_index'] = group_idx
                    detected_circles.append(circle)

        return output, detected_circles

    def create_circle_mask(self, image: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        return mask

    def group_circles(self, circles: List[Dict]) -> List[List[Dict]]:
        if not circles:
            return []

        circles = sorted(circles, key=lambda x: x['y'])
        groups = []
        current_group = [circles[0]]

        for circle in circles[1:]:
            if abs(circle['y'] - current_group[0]['y']) <= self.y_threshold:
                current_group.append(circle)
            else:
                current_group = sorted(current_group, key=lambda x: x['x'])
                groups.append(current_group)
                current_group = [circle]

        if current_group:
            current_group = sorted(current_group, key=lambda x: x['x'])
            groups.append(current_group)

        return groups

    def analyze_groups(self, groups: List[List[Dict]], answer_key: List[str]) -> Dict[str, int]:
        results = {
            'correct': 0,
            'incorrect': 0,
            'empty': 0,
            'invalid': 0
        }
        answer_status = []

        for i, group in enumerate(groups):
            if i >= len(answer_key):
                break

            marked_circles = sum(1 for circle in group if circle['is_filled'])

            if marked_circles == 0:
                results['empty'] += 1
                answer_status.append((i + 1, 'Empty', None, answer_key[i]))
            elif marked_circles > 1:
                results['invalid'] += 1
                answer_status.append((i + 1, 'Invalid', None, answer_key[i]))
            else:
                marked_index = next(i for i, circle in enumerate(group) if circle['is_filled'])
                marked_answer = string.ascii_lowercase[marked_index]
                correct_answer = answer_key[i]

                if marked_answer == correct_answer:
                    results['correct'] += 1
                    answer_status.append((i + 1, 'Correct', marked_answer, correct_answer))
                else:
                    results['incorrect'] += 1
                    answer_status.append((i + 1, 'Incorrect', marked_answer, correct_answer))

        return results, answer_status

    def print_results(self, results: Dict[str, int], answer_status: List[Tuple[int, str, str, str]], total_questions: int) -> None:
        print("\nResults:")
        print("Question | Marked Answer | Correct Answer | Status")
        for status in answer_status:
            marked_answer = status[2] if status[2] is not None else 'N'
            print(f"Q{status[0]} | {marked_answer} | {status[3]} | {status[1]}")

        accuracy = (results['correct'] / total_questions) * 100
        print(f"\nAccuracy: {accuracy:.2f}%")


def main(image_path: str, answer_key: List[str]):
    reader = OpticalFormReader()
    processed_image, circles = reader.detect_circles_and_marks(image_path, answer_key)
    groups = reader.group_circles(circles)
    results, answer_status = reader.analyze_groups(groups, answer_key)
    total_questions = len(answer_key)
    reader.print_results(results, answer_status, total_questions)

    cv2.imshow('Detected Circles', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "opticial.jpg"
    answer_key = ['a', 'c', 'a', 'd', 'e', 'e', 'b', 'a']
    main(image_path, answer_key)