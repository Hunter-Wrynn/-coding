#include <iostream>
using namespace std;

int a[101],n;//待排序数
void quickSort(int left,int right);
int main() {
	
	cin>>n;
	for(int i=1;i<=n;i++){
		cin>>a[i];
	}
	
	quickSort(1,n);
	
	for(int i=1;i<=n;i++){
		cout<<a[i]<<" ";
	}
	return 0;
}


void quickSort(int left,int right) {
	//递归结束条件为可排序块为0 
	if(left > right) {
		return;
	}
	//temp为选定基准 
	int temp = a[left];
	int i = left;
	int j = right;
	int t;
	while(i!=j) {
		//移动右指针，直到找到比基准小 
		while(i<j && a[j]>=temp) {
			j--;
		}
		//移动左指针，直到找到比基准大 
		while(i<j && a[i]<=temp) {
			i++;
		}
	
		//两指针未相遇时交换 
		if(i<j) {
			t = a[i];
			a[i] = a[j];
			a[j] = t;
		}
	}
	
	//将最终相遇点与基准交换 
	a[left] = a[i];
	a[i] = temp;
	
	//继续往左块和右块排序，直到最终为顺序集合 
	quickSort(left,i-1);
	quickSort(i+1,right);	
}
