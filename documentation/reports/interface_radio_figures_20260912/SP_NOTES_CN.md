# s/p 偏振：配图简短复习

先记住：**s 垂直入射面，p 平行入射面；二者都垂直光线。** 这里描述的是电场方向。

## 先确定“入射面”（PPT 第 4 页）

![入射面与 s/p 方向](figures/06_sp_geometry.png)

入射面由入射方向 \(\hat k_i\) 和界面法线 \(\hat N\) 张成。绿色 s 出纸面；紫色 p 在纸面内，并与对应的光线成直角。折射后光线转向，所以 p 的方向也要转。右图只是电场分解示意，箭头长度不是模拟场强。

代码采用：

$$
\hat s=\frac{\hat k_i\times\hat N}{|\hat k_i\times\hat N|},\qquad
\hat p_i=\hat s\times\hat k_i,\qquad
\hat p_t=\hat s\times\hat k_t.
$$

于是 \(E_s=\mathbf E_i\cdot\hat s\)、\(E_p=\mathbf E_i\cdot\hat p_i\)。s/p 常分别称为 TE/TM，但这里的“横向”是相对于入射面而言。它们不是一套固定的地理坐标；换一条光路或一个山面，基底可能改变。

## 界面对两个分量分别作用（PPT 第 5 页）

![s/p 透射系数](figures/07_sp_fresnel.png)

对于当前采用的各向同性、非磁性、无吸收界面，电场振幅透射系数为：

$$
t_s=\frac{2n_1\cos\theta_i}{n_1\cos\theta_i+n_2\cos\theta_t},\qquad
t_p=\frac{2n_1\cos\theta_i}{n_2\cos\theta_i+n_1\cos\theta_t}.
$$

功率还要计入折射率和穿过界面的角度：

$$
T_{s,p}=\frac{n_2\cos\theta_t}{n_1\cos\theta_i}|t_{s,p}|^2,\qquad R_{s,p}+T_{s,p}=1.
$$

所以 **t 大于 1 并不意味着增能**。这条真实光路的入射角是 7.107°，\(n_1=2\)、\(n_2=1.000244\)：

| 分量 | 振幅系数 t | 功率透射率 T |
| --- | ---: | ---: |
| s | 1.34379 | 88.18% |
| p | 1.35452 | 89.59% |

公式及功率定义参见 [TMM 算法说明](https://arxiv.org/abs/1603.02720)；图中的空心点来自实际安装的 TMM 0.2.0。角度扫描固定这条光路的局部折射率，是受控界面诊断，不代表额外的簇射事件。

几个容易记反的地方：

- **正入射**：s/p 等价，t_s = t_p；入射面本身不唯一，可以选任意一对横向正交基。
- **Brewster 角**：本例约 26.571°，消失的是 **p 反射**，不是 p 透射。
- **临界角**：从高 n 到低 n 才有，本例约 30.008°。更大角度发生全反射；仍可有倏逝场，但当前 beta5 不计算这种近界面场，也不计算反射波。图中灰色反射曲线仅为外部参考。
- **线偏振与椭圆偏振**：一般由 s/p 振幅和相对相位共同决定。本例透射系数为实数，界面只改变两分量的相对幅度，不额外引入相位差；有吸收、全反射相移或各向异性时不能照搬这一结论。

## 程序中怎样处理（PPT 第 6 页）

![真实轨迹 s/p 分量](figures/08_sp_actual_track.png)

取真实电子轨迹 1 的横向电流方向 \(q[\boldsymbol\beta-(\boldsymbol\beta\cdot\hat k_i)\hat k_i]\)，把其总长度归一化为 1。它的 s/p 因子约为 **0.7503 / 0.6611**。乘上各自的 Fresnel 系数后，得到 **1.0083 / 0.8954**；图中去除了两分量共用的几何扩散和通量归一化因子，所以这些数不是绝对电场，也不是功率份额。

右图直接调用 beta5 的矢量传递函数检查。局部 s/p 基底下矩阵为对角形式：各向同性界面不把 s 转成 p，但两者乘的系数不同。完整实现还包含公共因子：

$$
\mathbf E_{\rm out}\ \propto\
G\sqrt{\frac{n_2\cos\theta_t}{n_1\cos\theta_i}}
\left(t_s a_s\hat s+t_p a_p\hat p_t\right),
$$

这里 \(a_s,a_p\) 表示输入横向源因子的投影，\(G\) 包含几何扩散和振幅吸收。源的量纲、时间因子仍由 CoREAS/ZHS 发射模块处理；不能把该式当成孤立界面的平面波 Fresnel 关系重复乘通量因子。

实际叠加时先把每条光路的贡献转回共同的 ENU 坐标，再相干求和。**不能把不同光路各自的 s/p 数值直接当作同一组坐标相加。**

本次 684 个角度样本的 beta5/TMM 振幅系数最大相对差约 3.43e-12，功率透射率最大绝对差约 1.37e-11；局部传递矩阵检查误差约 6.66e-16。数据见 [sp_metrics.json](data/sp_metrics.json)，全部新增计算在 PSR 完成。

## 看完 s/p，先接 Ray tube（PPT 第 7–8 页）

![Ray tube definition](figures/12_raytube_concept.png)

Ray tube 是中心光线周围的一小束相邻光线，横截面垂直中心方向。在同一无吸收介质内，束功率不变，单位面积功率随 1/A⊥ 变化，电场幅度随 1/√A⊥ 变化。点源传播距离翻倍，截面积变为四倍，场幅减半。

面积 Jacobian **J=dA⊥/dΩᵢ** 把源点的小立体角映射成接收端的小截面积。均匀介质 J=R²；折射后要计算束的展开，不能一概用总路程代入 1/R。

![Ray-tube Jacobian comparison](figures/09_raytube.png)

误差图的颜色固定相对于界面法线的入射角；横轴为发射方向扰动步长 $h$，单位 rad，向右更小。纵轴为无量纲相对误差：

$$
\varepsilon_J(h)=\left|\frac{J_{\mathrm{RadioPropa}}(h)}{J_{\mathrm{beta5}}}-1\right|
$$

分子是 RadioPropa 的邻近射线中心差分估计，分母是 beta5 的解析参考；越低表示面积计算越一致。左图 29° 接近 30° 临界角，对方向扰动更敏感；右图最后的小平台与数值舍入影响相符。

8 组平面界面条件在 h=10⁻⁶ rad 时最大相对差约 3.79×10⁻¹⁰。这是几何量的对照，纵轴没有画信号强弱。完整的定义、h 的构造、临界角与平台解读见 [Ray tube 物理与读图](RAYTUBE_NOTES_CN.md)。

## 再看吸收衰减（PPT 第 9–10 页）

![Geometric spreading and amplitude absorption](figures/10_attenuation.png)

左图灰线为几何因子 1/R；蓝线加入吸收 exp(−R/L_E)，空心点为解析结果。右图核对均匀介质时延 nR/c。本页是 n=1.33、L_E=100 m 的控制算例；并非实际山体材料的折射率配置。

程序先对每条光路的**电场幅度**乘吸收因子。当前山体/空气两段路径使用：

$$
a_E=\exp\!\left(-\frac{\ell_{\rm rock}}{L_{E,\rm rock}}-\frac{\ell_{\rm air}}{L_{E,\rm air}}\right).
$$

ℓ 是射电光线在对应介质中的实际几何路程，单位米。单一介质直达路径则使用 exp(−ℓ/L_E)。这些计算位于 `Propagation.hpp::directPath` 和 `transmittedPath`；结果保存在 `Path::attenuation`，与 ray-tube 几何扩散和 Fresnel 因子一起进入电场传递矩阵，CoREAS/ZHS 共用这一传播实现。各条贡献分别传播后再相干叠加，完整波形不能套用某一条光路的统一衰减因子。

L_E 是**电场幅度的 1/e 长度**。同一介质中光强与电场幅度平方成正比；对同一条光路，比较吸收开/关、其余条件相同时：

$$
\frac{I_{\rm with\ absorption}}{I_{\rm without\ absorption}}=a_E^2,\qquad
I(\ell)/I_0=e^{-2\ell/L_E},\qquad L_I=L_E/2.
$$

以 SiO₂ 基准 L_E=100 m 为例，单独考虑吸收：

| 岩石内路程 | 电场剩余 | 光强剩余 |
| --- | ---: | ---: |
| 100 m | 36.8% | 13.5% |
| 200 m | 13.5% | 1.83% |
| 300 m | 4.98% | 0.248% |

上述 I(ℓ) 关系隔离了材料吸收，不包含点源几何扩散。在均匀介质中，完整场幅还含 1/R，完整光强还含 1/R²；跨界面时由 ray tube 和各自的 s/p Fresnel 系数处理。电场与光强的平方关系可参见 [COMSOL 吸收说明](https://doc.comsol.com/6.4/doc/com.comsol.help.roptics/roptics_ug_optics.6.68.html)。

当前实现采用每种介质一个常数 L_E：不随频率、位置或偏振改变，s/p 共用相同吸收因子。材料入口明确拒绝开启频率相关衰减或材料色散；尚未实现由复介电常数推导的频散吸收。山体应用的空气 L_E 保持默认无穷大，因此空气段吸收因子为 1，但其几何扩散、折射率和传播时延仍计算。岩石 L_E 由 `material.radio.field_attenuation_length_m` 传入，可由材料配置覆盖预设；这是指定的射电模型参数，程序不根据组分和密度自动推导该值。

**关于虚部折射率的局限：** 当前已有介质差异，体现在各材料使用不同的常数 L_E；缺少的是材料随频率变化的复电磁响应，以及体传播与界面传递的一致处理。

采用时间约定 exp(−iωt)，令复折射率为 ñ(f)=n′(f)+iκ(f)。均匀介质中的平面波传播因子为：

$$
\frac{E(f,\ell)}{E(f,0)}=\exp\!\left(i\frac{2\pi f}{c}n'(f)\ell\right)
\exp\!\left(-\frac{2\pi f}{c}\kappa(f)\ell\right),\qquad
L_E(f)=\frac{c}{2\pi f\kappa(f)}.
$$

所以，在体传播幅度这一层，当前常数 L_E 可以形式上对应 κ_eff(f)=c/(2πfL_E)，并非没有任何吸收。但程序没有将这一等效 κ 用于复数 Fresnel 系数，也没有把实部色散与吸收统一成完整材料响应。复折射率的约定与平面波公式见 [TMM 作者说明](https://arxiv.org/html/1603.02720v5#S2.SS1)。

下一层模型需要提供材料相关的 κ(f) 或等价的 L_E(f)，以及配套的 n′(f)：不同频率的吸收会改变信号频谱，实部色散会改变相位和脉冲形状，复数界面系数还会改变 s/p 振幅与相位。介质非均匀性、粗糙界面的散射等需要另外描述，单独加入 κ(f) 并不能覆盖这些效应。本段仅说明模型边界，尚未实施这些扩展。

**CoREAS/ZHS 对频率相关衰减的支持状态：** 两种发射表示都可以与按光路定义的线性频率响应结合；当前 beta5 山体实现尚未接入，不能通过开启配置开关直接使用。`TerrainMaterialConfig.hpp` 会拒绝 `frequency_dependent_attenuation_enabled: true`。

若先只扩展体吸收，保留当前实部折射率和几何光路近似，应在叠加各贡献时使用：

$$
\widetilde{\mathbf E}_{\rm obs}(f)=\sum_j\widetilde{\mathbf E}_{j,0}(f)
\exp\!\left[-\sum_m\frac{\ell_{jm}}{L_{E,m}(f)}\right].
$$

j 表示一个轨迹子段到该接收站的一条有效光路，m 表示经过的介质。Ẽⱼ,₀ 已包含该光路的源项、几何扩散、界面透射及到达相位，但不含旧常数吸收。这样可避免重复施加旧、新两套衰减。这个公式仅说明固定光路下的频率相关体吸收，不代表完整复折射率发射/传播模型已经实现。

CoREAS 的端点脉冲可以在频域乘相应传播响应，时域上等价于与该光路的响应核卷积；ZHS 的有限轨迹贡献也可以乘相同响应。两算法的发射表示可保留，传播接口与累加方式需要扩展。CORSIKA 8 将发射形式和传播器分开的设计可参见 [射电模块论文](https://arxiv.org/html/2409.15999v1)。

当前 `CoreasEndpoints.hpp` 和 `ZhsIntervals.hpp` 都先调用公共传递矩阵，再写入设备上的时间矩网格；网格按接收站、源介质、分量和时间归并，未保留每条光路的长度。`Output.cpp::render` 最后才对这些已合并的矩作 FFT。因此一般不能在最终总频谱上统一乘 exp(−ℓ/L_E(f))，山体源和空气源分成两组也仍不足以保留所有光程差异。只有各贡献确实共用同一传播响应时，统一滤波才严格成立。

实现时可在设备端逐频率累加，或引入经过误差控制的光程分组/响应展开，保留不同光路的权重；需要重新评估显存与计算量。当前输出的总波形、总频谱或已归并时间矩通常不足以精确补算任意 L_E(f)，若另存有完整轨迹和场景，可重放射电计算。上述方案尚未实施。

![Selected material attenuation lengths](figures/11_material_attenuation.png)

这张图只比较材料内路径的吸收。当前材料卡的 L_E：SiO₂ 100 m、CaCO₃ 14.48 m、花岗岩十元素混合物 86.86 m。吸收应按每条光路在各介质中的实际长度分别累计，不能给整场总波形统一套用一条路径的吸收，也不能重复乘已经计入的几何扩散。

## 然后看 E01 三分量时域波形（PPT 第 11 页）

![E01 three-component waveform](figures/04_E01_waveform.png)

先看左图的到达时段，再看右图约 14.18 μs 的主脉冲。三色是共同 ENU 坐标下的 East、North、Up 电场，单位 pV/m；各光路的局部 s/p 已分别转回 ENU，并按到达时间相干叠加。

这是同一 1 GeV 电子事例在 E01 的完整波形；上面的 s/p 图取自轨迹 1 到 `diagnostic_offset` 的光路，柱高不能直接换算成这里的绝对电场。左图虚线 11.192 μs 是所选空气光路的到达时刻，完整波形主峰不必与它重合。

随后看 [CoREAS/ZHS 对比图](figures/05_E01_coreas_zhs.png)，比较同一分量的峰位、幅度和复频谱差异。再进入高能事件的几何、shower profile、多站波形及来源分解；完整顺序已放到 [D 盘后续看图说明](/mnt/d/CorsikaData/beta5_doublebang_figures_20260914/AFTER_SP_CN.md)。
