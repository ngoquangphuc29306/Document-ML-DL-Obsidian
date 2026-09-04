---
tags: [optimization, genetic-algorithm, evolutionary-algorithms, machine-learning, ai]
aliases: [Thuật toán di truyền, Genetic Algorithm, GA]
date: 2026-09-04
---

# Genetic Algorithm (Thuật toán Di truyền)

## 1. Động lực: Tối ưu hóa Hộp đen (Black-box Optimization)

### 1.1. Hạn chế của các phương pháp xác định cổ điển
* **Phương pháp truyền thống:** Các giải thuật cổ điển (như tính đạo hàm trực tiếp $\frac{d}{dx}y(x) = 0$ hoặc Gradient Descent) yêu cầu hàm mục tiêu phải có công thức giải tích tường minh và khả vi.
* **Vấn đề bẫy cực trị địa phương (Multimodal Case):** Trên các không gian hàm mục tiêu có nhiều đỉnh đồi và thung lũng (multimodal landscape), các thuật toán dựa trên đạo hàm rất dễ bị rơi vào và mắc kẹt tại các điểm tối ưu cục bộ (**Local Optima / Suboptima**) thay vì tìm thấy nghiệm tối ưu toàn cục (**Global Optimum**).


![](../../../05_Assets/Images/Pasted%20image%2020260904204736.png)


### 1.2. Bản chất của Tối ưu hóa Hộp đen (Black-box Optimization)
* Trong nhiều bài toán thực tế, ta không có công thức toán học nội tại của hệ thống mà chỉ quan sát được quan hệ ánh xạ giữa đầu vào $x$ và đầu ra $f(x)$ thông qua mô phỏng hoặc đo đạc.
* **Evolutionary Algorithms (EAs):** Là nhóm thuật toán tìm kiếm và tối ưu hóa giải quyết bài toán thông qua việc mô phỏng các cơ chế tiến hóa sinh học tự nhiên mà không cần dựa vào đạo hàm hay gradient.

![Pasted image 20260904202248](../../../05_Assets/Images/Pasted%20image%2020260904202248.png)

---

## 2. Các Thành phần và Thuật ngữ Cốt lõi trong GA

| Thuật ngữ sinh học | Tương đương trong GA | Mô tả ý nghĩa toán học / kỹ thuật |
| :--- | :--- | :--- |
| **Gene (Gen)** | Biến / Thành phần đặc trưng | Đơn vị nhỏ nhất cấu thành nên giải pháp. Có thể mã hóa dạng nhị phân ($\{0, 1\}$), số nguyên ($\mathbb{N}$), hoặc số thực (Floating-point). |
| **Chromosome (Nhiễm sắc thể)** | Cá thể (Individual) / Vector nghiệm | Một chuỗi gồm $n$ gen. Đại diện cho **một nghiệm thử nghiệm** (candidate solution) trong không gian tìm kiếm $\Omega$. |
| **Population (Quần thể)** | Tập hợp nghiệm | Tập gồm $m$ cá thể cùng tồn tại và tiến hóa qua các thế hệ. |
| **Natural Selection** | Selection (Chọn lọc) | Cơ chế giữ lại và ưu tiên nhân bản các giải pháp chất lượng tốt ("Survival of the fittest"). |
| **Inheritance** | Crossover (Lai ghép) | Cơ chế tái tổ hợp các đoạn gen từ bố mẹ để sinh ra nghiệm con mới (Offspring). |
| **Mutation** | Mutation (Đột biến) | Biến đổi ngẫu nhiên giá trị của một số gen để mở rộng vùng tìm kiếm mới. |

![Pasted image 20260904202409](../../../05_Assets/Images/Pasted%20image%2020260904202409.png)

### Hàm thích nghi (Fitness Function)
Hàm thích nghi $f(x)$ lượng hóa mức độ đáp ứng mục tiêu của từng cá thể:
* **Bài toán Tối đa hóa (Maximization):** Giá trị mục tiêu càng cao thì fitness càng cao.
* **Bài toán Tối thiểu hóa (Minimization):** Đối với các hàm mất mát (loss), sai số (error), hoặc chi phí (cost) ký hiệu là $L(x)$, hàm fitness thường được chuyển đổi nghịch đảo để đảm bảo nghiệm có sai số càng nhỏ thì độ thích nghi càng lớn:
  $$\text{Fitness} = \frac{1}{L(x) + 1}$$

---

## 3. Quy trình Thực thi Chuẩn (Simple GA Pipeline)

Chu trình tiến hóa vận hành theo một vòng tuần hoàn lặp khép kín:
![Pasted image 20260904202527](../../../05_Assets/Images/Pasted%20image%2020260904202527.png)

### Bước 1: Population Initialization
Sinh ngẫu nhiên $m$ cá thể trong không gian tìm kiếm, mỗi cá thể gồm chuỗi $n$ gen:
```python
# Ví dụ khởi tạo chuỗi nhị phân độ dài n cho m cá thể
population = [[random.randint(0, 1) for _ in range(n)] for _ in range(m)]
```

### Bước 2: Fitness Evaluation
Áp dụng hàm đánh giá lên toàn bộ $m$ cá thể trong quần thể để gán giá trị thích ứng tương ứng cho từng cá thể.

### Bước 3: Selection / Reproduction
Lựa chọn các cá thể chất lượng để đưa vào nhóm phối giống (Mating Pool). Hai chiến lược phổ biến:
1. **Tournament Selection (Chọn lọc Đấu loại):** Chọn ngẫu nhiên hai hoặc nhiều cá thể từ quần thể, so sánh fitness giữa chúng và chọn cá thể chiến thắng.
2. **Roulette Wheel Selection:** Xác suất cá thể thứ $i$ được chọn tỷ lệ thuận với điểm fitness của chính nó:
   $$P_i = \frac{\text{Fitness}_i}{\sum_{j=1}^{m} \text{Fitness}_j}$$

![Pasted image 20260904202636](../../../05_Assets/Images/Pasted%20image%2020260904202636.png)

### Bước 4: Crossover / Recombination
Mô phỏng quá trình truyền mã di truyền từ hai phụ mẫu (Parent 1, Parent 2) để tạo ra các thế hệ con (Offspring). Quá trình này thực hiện khai thác các thông tin hữu ích sẵn có (**Exploitation**).
* **1-point Crossover:** Chọn 1 vị trí cắt ngẫu nhiên, hoán đổi đoạn gen phía sau điểm cắt giữa hai cá thể.
* **2-point / n-point Crossover:** Cắt tại 2 hoặc nhiều vị trí và hoán đổi các đoạn gen xen kẽ.
* **Uniform Crossover:** Duyệt qua từng gen, sinh số ngẫu nhiên $r \in [0, 1]$. Nếu $r < \text{rate}$, gen tại vị trí đó được hoán đổi giữa hai phụ mẫu.

![Pasted image 20260904202727](../../../05_Assets/Images/Pasted%20image%2020260904202727.png)

### Bước 5: Đột biến (Mutation)
Thực hiện thay đổi ngẫu nhiên giá trị của gen với xác suất nhỏ (thường từ $1\% - 5\%$).
* **Vai trò:** Đây là cơ chế khám phá không gian mới (**Exploration**). Nếu quần thể rơi vào bẫy cực trị địa phương và các cá thể trở nên tương đồng, thao tác đột biến sẽ giúp nhảy ra khỏi cực trị địa phương.

![Pasted image 20260904202932](../../../05_Assets/Images/Pasted%20image%2020260904202932.png)

---

## 4. Chiến lược Tinh hoa (Elitism Strategy)

> [!WARNING] Cảnh báo rủi ro suy thoái nghiệm
> Do cơ chế lai ghép và đột biến đều mang tính ngẫu nhiên, một cá thể có độ thích nghi xuất sắc nhất ở thế hệ hiện tại hoàn toàn có thể bị phá hủy cấu trúc gen tối ưu trong quá trình tiến hóa và biến thành một nghiệm kém hơn ở thế hệ kế tiếp.

* **Cơ chế Elitism:** Trực tiếp chọn lọc một số lượng nhỏ cá thể xuất sắc nhất (thường là $1 - 2$ cá thể dẫn đầu) từ thế hệ cha sang thẳng thế hệ con mà **không** áp dụng Crossover hay Mutation lên chúng.
* **Tác dụng:** Đảm bảo độ thích nghi lớn nhất của quần thể qua các thế hệ là một hàm đơn điệu tăng (hoặc ít nhất không bao giờ bị sụt giảm).
![Pasted image 20260904203056](../../../05_Assets/Images/Pasted%20image%2020260904203056.png)
---

## 5. Điều kiện Dừng Thuật toán (Termination Criteria)

Thuật toán kết thúc khi thỏa mãn một trong các tiêu chuẩn sau:
1. **Số thế hệ tối đa (Max Generations):** Đạt ngưỡng số vòng lặp `max_generations` định trước.
2. **Đạt ngưỡng mục tiêu:** Tìm được nghiệm có fitness đạt giá trị lý tưởng (ví dụ: chuỗi toàn bộ số 1 trong bài toán One-Max).
3. **Quần thể hội tụ (Convergence):** Độ đa dạng của quần thể suy giảm, độ lệch chuẩn fitness giữa các cá thể tiệm cận 0 hoặc giá trị fitness tốt nhất không cải thiện qua một số thế hệ nhất định (`stale generations`).

---

## 6. Bài toán Minh họa: One-Max Problem

* **Mô tả:** Tìm chuỗi nhị phân độ dài $n$ sao cho tổng các phần tử đạt giá trị cực đại.
* **Không gian tìm kiếm:** Có $2^n$ tổ hợp khả dĩ.
* **Hàm mục tiêu:** 
  $$\text{Fitness}(x) = \sum_{i=1}^{n} x_i$$
  Trong đó $x_i \in \{0, 1\}$. Cực đại đạt được khi chuỗi gồm toàn chữ số $1$ ($\text{Fitness} = n$).

### Cấu trúc Code mẫu (Python Native):
```python
import random
import matplotlib.pyplot as plt

def generate_random_value():
	return random.randint(0, 1)

def compute_fitness(individual):
    return sum(individual)

def create_individual(n):
    return [generate_random_value() for _ in range(n)]

def crossover(ind1, ind2, rate=0.9):
    ind1_new, ind2_new = ind1.copy(), ind2.copy()
    for i in range(len(ind1)):
        if random.random() < rate:
            ind1_new[i], ind2_new[i] = ind2[i], ind1[i]
    return ind1_new, ind2_new

def mutate(individual, rate=0.05):
    individual_m = individual.copy()
    for i in range(n):
        if random.random() < mutation_rate:
            individual_m[i] = generate_random()
    return individual_m

def selection(sorted_old_population, m):
    index1 = random.randint(0, m - 1)
    index2 = random.randint(0, m - 1)

    while index1 == index2:
        index2 = random.randint(0, m - 1)
    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]
    return individual_s

# Cấu hình tiến hóa
n = 20           # Chiều dài chuỗi gen
m = 40           # Kích thước quần thể
generations = 50 # Số thế hệ
elitism = 2      # Số lượng cá thể tinh hoa giữ lại

# để vẽ biểu đồ quá trình tối ưu
fitnesses = []

population = [create_individual(n) for _ in range(m)]

for gen in range(generations):
    # Sắp xếp quần thể giảm dần theo fitness
    population = sorted(population, key=compute_fitness, reverse=True)
    
    if i%1 == 0:
        fitnesses.append(compute_fitness(sorted_population[-1]))
        print("iter", i, "BEST:", compute_fitness(sorted_population[-1]))
    
    # Giữ lại tinh hoa
    new_population = population[:elitism]
    
    while len(new_population) < m:
        # Chọn lọc ngẫu nhiên từ nửa trên quần thể tốt
        p1 = selection(sorted_population, m)
        p2 = selection(sorted_population, m)
        
        # Lai ghép và đột biến
        c1, c2 = crossover(p1, p2)
        new_population.append(mutate(c1))
        if len(new_population) < m:
            new_population.append(mutate(c2))
            
    population = new_population

plt.figure(figsize=(10, 6))
plt.plot(fitnesses)
plt.title('Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()
```
